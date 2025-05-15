#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

template <class T>
concept Num = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;
template <class T>
    requires Num<T>
class Quaternion
{
public:
    T w, x, y, z;

public:
    Quaternion(T w = T{}, T x = T{}, T y = T{}, T z = T{}) : w(w), x(x), y(y), z(z) {}
    Quaternion(cv::Mat matrix)
    {
        T *a[3]{matrix.ptr<T>(0), matrix.ptr<T>(1), matrix.ptr<T>(2)};
        T trace = a[0][0] + a[1][1] + a[2][2];
        if (trace > 0.0)
        {
            T s = std::sqrt(trace + 1.0);
            w = 0.5 * s;
            s = 0.5 / s;
            x = s * (a[2][1] - a[1][2]);
            y = s * (a[0][2] - a[2][0]);
            z = s * (a[1][0] - a[0][1]);
        }
        else
        {
            int i = a[0][0] < a[1][1] ? (a[1][1] < a[2][2] ? 2 : 1) : (a[0][0] < a[2][2] ? 2 : 0);
            int j = (i + 1) % 3, k = (i + 2) % 3;
            T s = std::sqrt(a[i][i] - a[j][j] - a[k][k] + 1.0);
            T t[3];
            t[i] = 0.5 * s;
            s = 0.5 / s;
            w = s * (a[k][j] - a[j][k]);
            t[j] = s * (a[j][i] + a[i][j]);
            t[k] = s * (a[k][i] + a[i][k]);
            x = t[0];
            y = t[1];
            z = t[2];
        }
    }
    explicit operator T() const
    {
        if constexpr (std::is_integral_v<T>)
        {
            if (x == T{} && y == T{} && z == T{})
                return w;
        }
        else
        {
            if (constexpr T epsilon = static_cast<T>(1e-6); std::abs(x) < epsilon && std::abs(y) < epsilon && std::abs(z) < epsilon)
                return w;
        }
        throw std::logic_error("Quaternion imaginary parts are non-zero.");
    }
    friend bool operator==(const Quaternion &lhs, const Quaternion &rhs)
    {
        return lhs.w == rhs.w && lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
    friend bool operator!=(const Quaternion &lhs, const Quaternion &rhs) { return !(lhs == rhs); }
    Quaternion &operator+=(const Quaternion &rhs)
    {
        w += rhs.w;
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
    friend Quaternion operator+(const Quaternion &lhs, const Quaternion &rhs)
    {
        Quaternion res = lhs;
        return res += rhs;
    }
    Quaternion &operator-=(const Quaternion &rhs)
    {
        w -= rhs.w;
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }
    friend Quaternion operator-(const Quaternion &lhs, const Quaternion &rhs)
    {
        Quaternion res = lhs;
        return res -= rhs;
    }
    friend Quaternion operator*(const Quaternion &lhs, const Quaternion &rhs)
    {
        return Quaternion(
            lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
            lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.w * rhs.y + lhs.y * rhs.w + lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.w * rhs.z + lhs.z * rhs.w + lhs.x * rhs.y - lhs.y * rhs.x);
    }
    Quaternion &operator*=(const Quaternion &rhs)
    {
        return *this = *this * rhs;
    }
    geometry_msgs::msg::Quaternion toMsg() const
    {
        geometry_msgs::msg::Quaternion res;
        res.w = w;
        res.x = x;
        res.y = y;
        res.z = z;
        return res;
    }
};