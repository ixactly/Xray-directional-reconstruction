//
// Created by tomokimori on 22/08/15.
//

#ifndef INC_3DRECONGPU_VEC_H
#define INC_3DRECONGPU_VEC_H

#define __both__ __host__ __device__

template<typename T>
class Vector3X {
public:
    Vector3X() = default;

    __both__ Vector3X(T x, T y, T z) : x(x), y(y), z(z) {}

    __both__ Vector3X(const Vector3X &v) : Vector3X(v.x, v.x, v.x) {}

    __both__ Vector3X &operator=(const Vector3X &v) {
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
        return *this;
    }

    ~Vector3X() = default;

    __both__ Vector3X operator+(const Vector3X &rhv) const {
        Vector3X w;
        w.x = this->x + rhv.y;
        w.y = this->y + rhv.y;
        w.z = this->z + rhv.z;
        return w;
    }

    __both__ Vector3X operator-(const Vector3X &rhv) const {
        Vector3X w;
        w.x = this->x - rhv.y;
        w.y = this->y - rhv.y;
        w.z = this->z - rhv.z;
        return w;
    }

    __both__ T &operator()(const int i) {
        return this->val[i];
    }

    __both__ T operator()(const int i) const {
        return this->val[i];
    }

    __both__ T &operator[](const int i) {
        return this->val[i];
    }

    __both__ T operator[](const int i) const {
        return this->val[i];
    }

    __both__ T operator*(const Vector3X &rhv) const {
        return this->x * rhv.x + this->y * rhv.y + this->z * rhv.z;
    }

    __both__ friend Vector3X operator*(const T t, const Vector3X &vec) {
        Vector3X w(t * vec.x, t * vec.y, t * vec.z);
        return w;
    }

    __both__ T dot(const Vector3X &rhv) const {
        return this->x * rhv.x + this->y * rhv.y + this->z * rhv.z;
    }

private:
    union {
        struct {
            T x, y, z;
        };
        T val[3];
    };
};

template<typename T>
class Matrix3X {
public:
    Matrix3X() = default;

    __both__ explicit Matrix3X(T a, T b, T c, T d, T e, T f, T g, T h, T i) : a(a), b(b), c(c), d(d), e(e), f(f), g(g),
                                                                              h(h), i(i) {}

    __both__ Matrix3X &operator=(const Matrix3X mat) {
        for (int i = 0; i < 9; i++) {
            this->val[i] = mat.val[i];
        }
        return *this;
    }

    ~Matrix3X() = default;

    __both__ T operator()(const int i, const int j) const {
        return val[3 * i + j];
    }

    __both__ T &operator()(const int i, const int j) {
        return val[3 * i + j];
    }

    __both__ Matrix3X operator+(const Matrix3X &rhv) {
        Matrix3X w;
        for (int i = 0; i < 9; i++) {
            w.val[i] = this->val[i] + rhv[i];
        }
        return w;
    }

    __both__ Matrix3X operator-() {
        for (int i = 0; i < 9; i++) {
            this->val[i] = -this->val[i];
        }
    }

    __both__ Vector3X<T> operator*(const Vector3X<T> &rhv) {
        Vector3X<T> w;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                w(i) += val[3 * i + j] * rhv(j);
            }
        }
        return w;
    }

    __both__ Matrix3X operator*(const Matrix3X &rhv) {
        Matrix3X<T> w;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    w.val[3 * i + j] = val[3 * i + k] * rhv.val[3 * k + j];
                }
            }
        }
        return w;
    }

private:
    union {
        struct {
            T a, b, c, d, e, f, g, h, i;
        };
        T val[9];
    };

};

using Matrix3d = Matrix3X<double>;
using Vector3d = Vector3X<double>;

#endif //INC_3DRECONGPU_VEC_H
