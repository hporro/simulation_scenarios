#pragma once

__constant__ static const unsigned int B[] = { 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF };
__constant__ static const unsigned int S[] = { 1, 2, 4, 8 };

__device__ unsigned int morton_code(unsigned int x, unsigned int y) {

    unsigned int a = x;
    unsigned int b = y;

    x = (x | (x << S[3])) & B[3];
    x = (x | (x << S[2])) & B[2];
    x = (x | (x << S[1])) & B[1];
    x = (x | (x << S[0])) & B[0];

    y = (y | (y << S[3])) & B[3];
    y = (y | (y << S[2])) & B[2];
    y = (y | (y << S[1])) & B[1];
    y = (y | (y << S[0])) & B[0];

    //printf("x: %d y: %d z: %d\n", a, b, (x | (y << 1)));

    return x | (y << 1);
}
