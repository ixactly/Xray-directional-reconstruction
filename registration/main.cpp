#include <math.h>
#include <omp.h>
#include <stdio.h>

#include "SVD.h" //特異値分解のライブラリ（numerical recipie より引用）

inline void computeRigid(double R[3][3], double T[3], int M, float (*p)[3], float (*q)[3]) {
    // 重心と誤差の計算
    double cp[3] = {0, 0, 0};
    double cq[3] = {0, 0, 0};
    for(int i = 0; i < M; i++) {
        cp[0] += p[i][0];
        cp[1] += p[i][1];
        cp[2] += p[i][2];
        cq[0] += q[i][0];
        cq[1] += q[i][1];
        cq[2] += q[i][2];
    }
    cp[0] /= M;
    cp[1] /= M;
    cp[2] /= M;
    cq[0] /= M;
    cq[1] /= M;
    cq[2] /= M;

    double **U, *S, **V; // 特異値分解用の変数
    U = new double*[4];
    S = new double[4];
    V = new double*[4];
    for(int i = 1; i <= 3; i++) {
        U[i] = new double[4];
        V[i] = new double[4];
    }

    // 剛体変換の計算: 特異値分解で計算する方法
    for(int i = 1; i <= 3; i++)
        for(int j = 1; j <= 3; j++)
            U[i][j] = 0;

    for(int i = 0; i < M; i++) {
        U[1][1] += (q[i][0] - cq[0]) * (p[i][0] - cp[0]);
        U[1][2] += (q[i][0] - cq[0]) * (p[i][1] - cp[1]);
        U[1][3] += (q[i][0] - cq[0]) * (p[i][2] - cp[2]);
        U[2][1] += (q[i][1] - cq[1]) * (p[i][0] - cp[0]);
        U[2][2] += (q[i][1] - cq[1]) * (p[i][1] - cp[1]);
        U[2][3] += (q[i][1] - cq[1]) * (p[i][2] - cp[2]);
        U[3][1] += (q[i][2] - cq[2]) * (p[i][0] - cp[0]);
        U[3][2] += (q[i][2] - cq[2]) * (p[i][1] - cp[1]);
        U[3][3] += (q[i][2] - cq[2]) * (p[i][2] - cp[2]);
    }

    svdcmp(U, 3, 3, S, V);

    // 回転行列
    R[0][0] = U[1][1] * V[1][1] + U[1][2] * V[1][2] + U[1][3] * V[1][3];
    R[0][1] = U[1][1] * V[2][1] + U[1][2] * V[2][2] + U[1][3] * V[2][3];
    R[0][2] = U[1][1] * V[3][1] + U[1][2] * V[3][2] + U[1][3] * V[3][3];
    R[1][0] = U[2][1] * V[1][1] + U[2][2] * V[1][2] + U[2][3] * V[1][3];
    R[1][1] = U[2][1] * V[2][1] + U[2][2] * V[2][2] + U[2][3] * V[2][3];
    R[1][2] = U[2][1] * V[3][1] + U[2][2] * V[3][2] + U[2][3] * V[3][3];
    R[2][0] = U[3][1] * V[1][1] + U[3][2] * V[1][2] + U[3][3] * V[1][3];
    R[2][1] = U[3][1] * V[2][1] + U[3][2] * V[2][2] + U[3][3] * V[2][3];
    R[2][2] = U[3][1] * V[3][1] + U[3][2] * V[3][2] + U[3][3] * V[3][3];

    // 平行移動ベクトル
    T[0] = cq[0] - (R[0][0] * cp[0] + R[0][1] * cp[1] + R[0][2] * cp[2]);
    T[1] = cq[1] - (R[1][0] * cp[0] + R[1][1] * cp[1] + R[1][2] * cp[2]);
    T[2] = cq[2] - (R[2][0] * cp[0] + R[2][1] * cp[1] + R[2][2] * cp[2]);

    for(int i = 1; i <= 3; i++) {
        delete[] U[i];
        delete[] V[i];
    }
    delete[] U;
    delete[] S;
    delete[] V;
}

// 二つのSTLファイルを読込み位置合わせの行列を計算する
void computeTrans(double R[3][3], double T[3], char* pathC, char* pathR) {
    int M; // 三角形の数
    FILE* in = fopen(pathC, "rb");
    fseek(in, 80, SEEK_SET);
    fread(&M, 4, 1, in);
    float(*p)[3] = new float[M][3];
    for(int i = 0; i < M; i++) {
        float a[3], b[3], c[3];
        float n[3];       // ダミー
        unsigned short d; // ダミー
        fread(n, 4, 3, in);
        fread(a, 4, 3, in);
        fread(b, 4, 3, in);
        fread(c, 4, 3, in);
        fread(&d, 2, 1, in);
        p[i][0] = (a[0] + b[0] + c[0]) / 3;
        p[i][1] = (a[1] + b[1] + c[1]) / 3;
        p[i][2] = (a[2] + b[2] + c[2]) / 3;
    }
    fclose(in);

    in = fopen(pathR, "rb");
    fseek(in, 84, SEEK_SET);
    float(*q)[3] = new float[M][3];
    for(int i = 0; i < M; i++) {
        float a[3], b[3], c[3];
        float n[3];       // ダミー
        unsigned short d; // ダミー
        fread(n, 4, 3, in);
        fread(a, 4, 3, in);
        fread(b, 4, 3, in);
        fread(c, 4, 3, in);
        fread(&d, 2, 1, in);
        q[i][0] = (a[0] + b[0] + c[0]) / 3;
        q[i][1] = (a[1] + b[1] + c[1]) / 3;
        q[i][2] = (a[2] + b[2] + c[2]) / 3;
    }
    fclose(in);

    computeRigid(R, T, M, p, q);

    delete[] p;
    delete[] q;
}

// 第1引数 の STL から 第2引数の STL への変換
int main(int argc, char* argv[]) {
    double R[3][3], T[3];
    
    char* pathC = "C:\\Users\\m1411\\source\\repos\\Xray-directional-reconstruction\\registration\\gfrp_at_iter15_ir2_500x500x500_shift.stl";
    char* pathR = "C:\\Users\\m1411\\source\\repos\\Xray-directional-reconstruction\\registration\\gfrp_at_iter15_ir2_500x500x500_rot.stl";
    computeTrans(R, T, pathC, pathR);

    printf("%ff, %ff, %ff, \n", R[0][0], R[0][1], R[0][2]);
    printf("%ff, %ff, %ff, \n", R[1][0], R[1][1], R[1][2]);
    printf("%ff, %ff, %ff, \n", R[2][0], R[2][1], R[2][2]);
    printf("%ff, %ff, %ff,", T[0], T[1], T[2]);

    /*
    (x, y, z) を （x1, y1, z1) に変換する場合
    x1 = R[0][0]*x + R[0][1]*y + R[0][2]*z + T[0];
    y1 = R[1][0]*x + R[1][1]*y + R[1][2]*z + T[1];
    z1 = R[2][0]*x + R[2][1]*y + R[2][2]*z + T[2];

    逆
    x = R[0][0]*(x-T[0]) + R[1][0]*(y-T[1]) + R[2][0]*(z-T[2])
    y = R[0][1]*(x-T[0]) + R[1][1]*(y-T[1]) + R[2][1]*(z-T[2]);
    z = R[0][2]*(x-T[0]) + R[1][2]*(y-T[1]) + R[2][2]*(z-T[2]);
    */

    return 0;
}
