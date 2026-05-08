
export const MathUtils = {
    identity: () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
    multiply: (a: Float32Array, b: Float32Array) => {
        const out = new Float32Array(16);
        let a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
        let a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
        let a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
        let a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

        let b0  = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
        out[0] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
        out[1] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
        out[2] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
        out[3] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

        b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
        out[4] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
        out[5] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
        out[6] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
        out[7] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

        b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
        out[8] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
        out[9] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
        out[10] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
        out[11] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

        b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
        out[12] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
        out[13] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
        out[14] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
        out[15] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
        return out;
    },
    fromTRS: (t: number[], r: number[], s: number[]) => {
        const out = new Float32Array(16);
        const x = r[0], y = r[1], z = r[2], w = r[3];
        const x2 = x + x, y2 = y + y, z2 = z + z;
        const xx = x * x2, xy = x * y2, xz = x * z2;
        const yy = y * y2, yz = y * z2, zz = z * z2;
        const wx = w * x2, wy = w * y2, wz = w * z2;

        out[0] = (1 - (yy + zz)) * s[0]; out[1] = (xy + wz) * s[0]; out[2] = (xz - wy) * s[0]; out[3] = 0;
        out[4] = (xy - wz) * s[1]; out[5] = (1 - (xx + zz)) * s[1]; out[6] = (yz + wx) * s[1]; out[7] = 0;
        out[8] = (xz + wy) * s[2]; out[9] = (yz - wx) * s[2]; out[10]= (1 - (xx + yy)) * s[2]; out[11]= 0;
        out[12]= t[0]; out[13]= t[1]; out[14]= t[2]; out[15]= 1;
        return out;
    },
    transformPos: (v: number[], m: Float32Array) => {
        const x = v[0], y = v[1], z = v[2];
        const w = m[3] * x + m[7] * y + m[11] * z + m[15] || 1.0;
        return [
            (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
            (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
            (m[2] * x + m[6] * y + m[10] * z + m[14]) / w
        ];
    },
    transformNorm: (v: number[], m: Float32Array) => {
        const x = v[0], y = v[1], z = v[2];
        return [
            m[0] * x + m[4] * y + m[8] * z,
            m[1] * x + m[5] * y + m[9] * z,
            m[2] * x + m[6] * y + m[10] * z
        ];
    },
    lerp: (a: number[], b: number[], t: number, out: Float32Array, offset: number) => {
        out[offset] = a[0] + t * (b[0] - a[0]);
        out[offset + 1] = a[1] + t * (b[1] - a[1]);
        out[offset + 2] = a[2] + t * (b[2] - a[2]);
    },
    slerp: (a: number[], b: number[], t: number, out: Float32Array, offset: number) => {
        // Standard Quaternion Slerp
        let cosHalfTheta = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        let bX = b[0], bY = b[1], bZ = b[2], bW = b[3];

        // If cosHalfTheta < 0, negate one quat to take the shortest path
        if (cosHalfTheta < 0) {
            bX = -bX; bY = -bY; bZ = -bZ; bW = -bW;
            cosHalfTheta = -cosHalfTheta;
        }

        if (Math.abs(cosHalfTheta) >= 1.0) {
            out[offset] = a[0]; out[offset+1] = a[1]; out[offset+2] = a[2]; out[offset+3] = a[3]; return;
        }

        const halfTheta = Math.acos(cosHalfTheta);
        const sinHalfTheta = Math.sqrt(1.0 - cosHalfTheta * cosHalfTheta);

        if (Math.abs(sinHalfTheta) < 0.001) {
            out[offset] = (a[0] * 0.5 + bX * 0.5); out[offset+1] = (a[1] * 0.5 + bY * 0.5);
            out[offset+2] = (a[2] * 0.5 + bZ * 0.5); out[offset+3] = (a[3] * 0.5 + bW * 0.5); return;
        }

        const ratioA = Math.sin((1 - t) * halfTheta) / sinHalfTheta;
        const ratioB = Math.sin(t * halfTheta) / sinHalfTheta;

        out[offset] = (a[0] * ratioA + bX * ratioB); out[offset+1] = (a[1] * ratioA + bY * ratioB);
        out[offset+2] = (a[2] * ratioA + bZ * ratioB); out[offset+3] = (a[3] * ratioA + bW * ratioB);
    }
};