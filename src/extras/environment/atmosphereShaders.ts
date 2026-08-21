import type { AtmosphereConfig } from './atmosphereMath';
import { resolveAtmosphereConfig } from './atmosphereMath';
import type { ShaderSkyLayout } from './types';

function n(x: number): string { const s = String(x); return /[.e]/i.test(s) ? s : `${s}.0`; }

/**
 * The medium's constants as WGSL. Exported because Phase 5's froxel kernel marches the SAME
 * medium and must not re-derive it — a second copy of these numbers is a drift surface, and the
 * whole point of §6's unification is that there is one medium (environment-aerial-design.md §6.1).
 */
export function buildAtmosphereConstantsWGSL(config: AtmosphereConfig = {}): string {
    return constants(config);
}

function constants(config: AtmosphereConfig): string {
    const c = resolveAtmosphereConfig(config);
    return /* wgsl */ `
const ATM_GROUND: f32 = ${n(c.groundRadius)};
const ATM_TOP: f32 = ${n(c.atmosphereTopRadius)};
const ATM_RAYLEIGH: vec3<f32> = vec3<f32>(${c.rayleighScattering.map(n).join(', ')});
const ATM_MIE_SCATTER: f32 = ${n(c.mieScattering)};
const ATM_MIE_EXTINCT: f32 = ${n(c.mieScattering + c.mieAbsorption)};
const ATM_OZONE: vec3<f32> = vec3<f32>(${c.ozoneAbsorption.map(n).join(', ')});
const ATM_HR: f32 = ${n(c.rayleighScaleHeight)};
const ATM_HM: f32 = ${n(c.mieScaleHeight)};
const ATM_OZONE_CENTRE: f32 = ${n(c.ozoneCentre)};
const ATM_OZONE_HALF: f32 = ${n(c.ozoneHalfWidth)};
const ATM_G: f32 = ${n(c.mieG)};
`;
}

/**
 * Density profiles, sphere intersection, the transmittance parameterization, and the phase
 * functions. Exported for the same reason as {@link buildAtmosphereConstantsWGSL}: Phase 5's
 * froxel kernel needs `atm_transUv` and `atm_sphereDistance` verbatim, and a re-implementation
 * would be a second kernel to keep in step (design-principles §13.1).
 */
export const ATMOSPHERE_MATH_WGSL = /* wgsl */ `
fn atm_densityR(h: f32) -> f32 { return exp(-max(h, 0.0) / ATM_HR); }
fn atm_densityM(h: f32) -> f32 { return exp(-max(h, 0.0) / ATM_HM); }
fn atm_densityO(h: f32) -> f32 { return max(0.0, 1.0 - abs(h - ATM_OZONE_CENTRE) / ATM_OZONE_HALF); }
fn atm_sphereDistance(radius: f32, r: f32, mu: f32) -> f32 {
    let b = r * mu; let q = b * b - r * r + radius * radius;
    if (q < 0.0) { return -1.0; }
    let s = sqrt(q); let near = -b - s; let far = -b + s;
    return select(select(select(-1.0, 0.0, near >= 0.0 || far >= 0.0), far, far > 1.0e-6), near, near > 1.0e-6);
}
fn atm_boundaryDistance(h: f32, mu: f32) -> f32 {
    let r = ATM_GROUND + max(h, 0.0);
    let top = atm_sphereDistance(ATM_TOP, r, mu);
    let ground = atm_sphereDistance(ATM_GROUND, r, mu);
    return select(top, ground, ground > 1.0e-6 && ground < top);
}
fn atm_transUv(h: f32, mu: f32) -> vec2<f32> {
    let r = ATM_GROUND + max(h, 0.0);
    let H = sqrt(ATM_TOP * ATM_TOP - ATM_GROUND * ATM_GROUND);
    let rho = sqrt(max(0.0, r * r - ATM_GROUND * ATM_GROUND));
    let d = atm_sphereDistance(ATM_TOP, r, mu);
    return clamp(vec2<f32>((d - (ATM_TOP-r)) / max(rho + H - (ATM_TOP-r), 1.0e-6), rho / H), vec2<f32>(0.0), vec2<f32>(1.0));
}
fn atm_decodeTransUv(uv: vec2<f32>) -> vec2<f32> {
    let H = sqrt(ATM_TOP * ATM_TOP - ATM_GROUND * ATM_GROUND);
    let rho = uv.y * H; let r = sqrt(rho * rho + ATM_GROUND * ATM_GROUND);
    let d0 = ATM_TOP-r; let d = d0 + uv.x * (rho + H - d0);
    let mu = select(1.0, clamp((ATM_TOP*ATM_TOP-r*r-d*d)/(2.0*r*d), -1.0, 1.0), d > 0.0);
    return vec2<f32>(r - ATM_GROUND, mu);
}
fn atm_phaseR(c: f32) -> f32 { return 3.0 * (1.0 + c*c) / (16.0 * 3.14159265359); }
fn atm_phaseM(c: f32) -> f32 {
    let g2 = ATM_G * ATM_G;
    return 3.0*(1.0-g2)*(1.0+c*c)/(8.0*3.14159265359*(2.0+g2)*pow(1.0+g2-2.0*ATM_G*c, 1.5));
}
`;

export function buildTransmittanceWGSL(config: AtmosphereConfig = {}): string {
    const c = resolveAtmosphereConfig(config);
    return /* wgsl */ `
${constants(c)}${ATMOSPHERE_MATH_WGSL}
@group(0) @binding(0) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8,8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(dst); if (any(id.xy >= dim)) { return; }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dim);
    let hm = atm_decodeTransUv(uv); let r0 = ATM_GROUND + hm.x;
    let groundHit = atm_sphereDistance(ATM_GROUND, r0, hm.y);
    if (groundHit > 1.0e-6) { textureStore(dst, vec2<i32>(id.xy), vec4<f32>(0.0,0.0,0.0,1.0)); return; }
    let distance = atm_boundaryDistance(hm.x, hm.y); let ds = distance / ${n(c.transmittanceSteps)};
    var od = vec3<f32>(0.0);
    for (var i=0u; i<=${c.transmittanceSteps}u; i=i+1u) {
        let s=ds*f32(i); let h=sqrt(max(0.0,r0*r0+s*s+2.0*r0*hm.y*s))-ATM_GROUND;
        let w=select(1.0,0.5,i==0u || i==${c.transmittanceSteps}u);
        od += (ATM_RAYLEIGH*atm_densityR(h)+vec3<f32>(ATM_MIE_EXTINCT*atm_densityM(h))+ATM_OZONE*atm_densityO(h))*w;
    }
    textureStore(dst, vec2<i32>(id.xy), vec4<f32>(exp(-od*ds*1000.0),1.0));
}`;
}

export function buildMultipleScatteringWGSL(config: AtmosphereConfig = {}): string {
    const c = resolveAtmosphereConfig(config);
    return /* wgsl */ `
${constants(c)}${ATMOSPHERE_MATH_WGSL}
@group(0) @binding(0) var trans: texture_2d<f32>;
@group(0) @binding(1) var lutSampler: sampler;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
var<workgroup> partial: array<vec3<f32>,64>;
@compute @workgroup_size(64)
fn cs_main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let dim=textureDimensions(dst); if (group.x>=dim.x || group.y>=dim.y) { return; }
    let uv=(vec2<f32>(group.xy)+0.5)/vec2<f32>(dim); let h=uv.y*(ATM_TOP-ATM_GROUND);
    let muS=uv.x*2.0-1.0; let z=1.0-2.0*(f32(t)+0.5)/64.0;
    let phi=6.28318530718*fract(f32(t)*0.61803398875); let q=sqrt(max(0.0,1.0-z*z));
    let dir=vec3<f32>(q*cos(phi),z,q*sin(phi)); let distance=atm_boundaryDistance(h,dir.y);
    let ds=distance/${n(c.multipleScatteringSteps)}; var sum=vec3<f32>(0.0); let r0=ATM_GROUND+h;
    for(var i=0u;i<${c.multipleScatteringSteps}u;i=i+1u){
        let s=(f32(i)+0.5)*ds; let r=sqrt(max(0.0,r0*r0+s*s+2.0*r0*dir.y*s)); let hh=r-ATM_GROUND;
        let muLocal=(r0*dir.y+s)/max(r,1.0); let T=textureSampleLevel(trans,lutSampler,atm_transUv(hh,muLocal),0.0).rgb;
        let source=ATM_RAYLEIGH*atm_densityR(hh)+vec3<f32>(ATM_MIE_SCATTER*atm_densityM(hh));
        sum += T*source*ds*1000.0;
    }
    let sunVisible=select(1.0,0.0,atm_sphereDistance(ATM_GROUND,ATM_GROUND+h,muS)>1.0e-6);
    let sunT=textureSampleLevel(trans,lutSampler,atm_transUv(h,muS),0.0).rgb*sunVisible;
    partial[t]=sum*sunT; workgroupBarrier();
    for(var stride=32u;stride>0u;stride=stride>>1u){ if(t<stride){partial[t]+=partial[t+stride];} workgroupBarrier(); }
    if(t==0u){ let psi=partial[0]/64.0; textureStore(dst,vec2<i32>(group.xy),vec4<f32>(psi/(vec3<f32>(1.0)-min(psi,vec3<f32>(0.95))),1.0)); }
}`;
}

export function buildSkyViewWGSL(config: AtmosphereConfig = {}): string {
    const c = resolveAtmosphereConfig(config);
    return /* wgsl */ `
${constants(c)}${ATMOSPHERE_MATH_WGSL}
struct Params { sunDir: vec3<f32>, cameraAltitude: f32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var trans: texture_2d<f32>;
@group(0) @binding(2) var multi: texture_2d<f32>;
@group(0) @binding(3) var lutSampler: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8,4)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>){
    let dim=textureDimensions(dst); if(any(id.xy>=dim)){return;} let uv=(vec2<f32>(id.xy)+0.5)/vec2<f32>(dim);
    let az=(uv.x-0.5)*6.28318530718; let x=uv.y*2.0-1.0; let horizon=sign(x)*x*x*1.57079632679;
    let view=vec3<f32>(cos(horizon)*cos(az),sin(horizon),cos(horizon)*sin(az));
    let h=params.cameraAltitude; let r0=ATM_GROUND+h; let distance=atm_boundaryDistance(h,view.y); let ds=distance/${n(c.skyViewSteps)};
    let sunTo=-normalize(params.sunDir); var L=vec3<f32>(0.0); var Tview=vec3<f32>(1.0);
    for(var i=0u;i<${c.skyViewSteps}u;i=i+1u){
        let s=(f32(i)+0.5)*ds; let p=vec3<f32>(0.0,r0,0.0)+view*s; let r=length(p); let hh=r-ATM_GROUND; let up=p/r;
        let muV=dot(view,up); let muS=dot(sunTo,up); let ext=ATM_RAYLEIGH*atm_densityR(hh)+vec3<f32>(ATM_MIE_EXTINCT*atm_densityM(hh))+ATM_OZONE*atm_densityO(hh);
        let sunVisible=select(1.0,0.0,atm_sphereDistance(ATM_GROUND,r,muS)>1.0e-6);
        let Ts=textureSampleLevel(trans,lutSampler,atm_transUv(hh,muS),0.0).rgb*sunVisible; let psi=textureSampleLevel(multi,lutSampler,vec2<f32>(muS*0.5+0.5,hh/(ATM_TOP-ATM_GROUND)),0.0).rgb;
        let c=dot(view,sunTo); let scatter=ATM_RAYLEIGH*atm_densityR(hh)*atm_phaseR(c)+vec3<f32>(ATM_MIE_SCATTER*atm_densityM(hh)*atm_phaseM(c));
        L += Tview*(Ts*scatter+psi*(ATM_RAYLEIGH*atm_densityR(hh)+vec3<f32>(ATM_MIE_SCATTER*atm_densityM(hh))))*ds*1000.0;
        Tview *= exp(-ext*ds*1000.0);
    }
    textureStore(dst,vec2<i32>(id.xy),vec4<f32>(max(L,vec3<f32>(0.0)),1.0));
}`;
}

export function buildAtmosphereSkyLayout(base: number, group: number): ShaderSkyLayout {
    const P=(o:number)=>`skyParams[${base+o}]`;
    return { wgslStructs: /* wgsl */ `
@group(${group}) @binding(14) var sky_atm_trans: texture_2d<f32>;
@group(${group}) @binding(15) var sky_atm_view: texture_2d<f32>;
@group(${group}) @binding(16) var sky_atm_sampler: sampler;`, wgslLogic: /* wgsl */ `
fn sky_atm_uv(dir: vec3<f32>) -> vec2<f32>{
    let az=atan2(dir.z,dir.x)/6.28318530718+0.5; let a=asin(clamp(dir.y,-1.0,1.0));
    return vec2<f32>(fract(az),0.5+0.5*sign(a)*sqrt(abs(a)/1.57079632679));
}
fn skySunRadiance(elevDeg:f32)->vec3<f32>{
    return textureSampleLevel(sky_atm_trans,sky_atm_sampler,vec2<f32>(${P(8)},${P(9)}),0.0).rgb*${P(6)};
}
fn sampleSky(dir:vec3<f32>)->vec3<f32>{
    var L=textureSampleLevel(sky_atm_view,sky_atm_sampler,sky_atm_uv(dir),0.0).rgb;
    let sunTo=vec3<f32>(-${P(0)},-${P(1)},-${P(2)}); let d=dot(dir,sunTo);
    let disc=smoothstep(${P(3)}-${P(4)},${P(3)}+${P(4)},d);
    return L+disc*${P(5)}*skySunRadiance(degrees(asin(clamp(sunTo.y,-1.0,1.0))));
}`};
}

export function buildAtmosphereBakeLayout(base: number): ShaderSkyLayout {
    const layout=buildAtmosphereSkyLayout(base,0);
    return { wgslStructs: layout.wgslStructs.replace(/binding\(14\)/g,'binding(3)').replace(/binding\(15\)/g,'binding(4)').replace(/binding\(16\)/g,'binding(5)'), wgslLogic: layout.wgslLogic };
}