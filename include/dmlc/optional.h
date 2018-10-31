/*!
 * Copyright (c) 2016 by Contributors
 * \file optional.h
 * \brief Container to hold optional data.
 */
#ifndef DMLC_OPTIONAL_H_
#define DMLC_OPTIONAL_H_

#include <iostream>
#include <string>
#include <utility>
#include <algorithm>

#include "./base.h"
#include "./common.h"
#include "./logging.h"
#include "./type_traits.h"

namespace dmlc {

/*! \brief dummy type for assign null to optional */
struct nullopt_t {
#if defined(_MSC_VER) && _MSC_VER < 1900
  /*! \brief dummy constructor */
  explicit nullopt_t(int a) {}
#else
  /*! \brief dummy constructor */
  constexpr nullopt_t(int a) {}
#endif
};

/*! Assign null to optional: optional<T> x = nullopt; */
constexpr const nullopt_t nullopt = nullopt_t(0);

/*!
 * \brief c++17 compatible optional class.
 *
 * At any time an optional<T> instance either
 * hold no value (string representation "None")
 * or hold a value of type T.
 */
template<typename T>
class optional {
 public:
  /*! \brief construct an optional object that contains no value */
  optional() : is_none(true) {}
  /*! \brief construct an optional object with value */
  explicit optional(const T& value) {
    is_none = false;
    new (&val) T(value);
  }
  /*! \brief construct an optional object with another optional object */
  optional(const optional<T>& other) {
    is_none = other.is_none;
    if (!is_none) {
      new (&val) T(other.value());
    }
  }
  /*! \brief deconstructor */
  ~optional() {
    if (!is_none) {
      reinterpret_cast<T*>(&val)->~T();
    }
  }
  /*! \brief swap two optional */
  void swap(optional<T>& other) {
    std::swap(val, other.val);
    std::swap(is_none, other.is_none);
  }
  /*! \brief set this object to hold value
   *  \param value the value to hold
   *  \return return self to support chain assignment
   */
  optional<T>& operator=(const T& value) {
    (optional<T>(value)).swap(*this);
    return *this;
  }
  /*! \brief set this object to hold the same value with other
   *  \param other the other object
   *  \return return self to support chain assignment
   */
  optional<T>& operator=(const optional<T> &other) {
    (optional<T>(other)).swap(*this);
    return *this;
  }
  /*! \brief clear the value this object is holding.
   *         optional<T> x = nullopt;
   */
  optional<T>& operator=(nullopt_t) {
    (optional<T>()).swap(*this);
    return *this;
  }
  /*! \brief non-const dereference operator */
  T& operator*() {  // NOLINT(*)
    return *reinterpret_cast<T*>(&val);
  }
  /*! \brief const dereference operator */
  const T& operator*() const {
    return *reinterpret_cast<const T*>(&val);
  }
  /*! \brief equal comparison */
  bool operator==(const optional<T>& other) const {
    return this->is_none == other.is_none &&
           (this->is_none == true || this->value() == other.value());
  }
  /*! \brief return the holded value.
   *         throws std::logic_error if holding no value
   */
  const T& value() const {
    if (is_none) {
      throw std::logic_error("bad optional access");
    }
    return *reinterpret_cast<const T*>(&val);
  }
  /*! \brief whether this object is holding a value */
  explicit operator bool() const { return !is_none; }
  /*! \brief whether this object is holding a value (alternate form). */
  bool has_value() const { return operator bool(); }

 private:
  // whether this is none
  bool is_none;
  // on stack storage of value
  typename std::aligned_storage<sizeof(T), alignof(T)>::type val;
};

/*! \brief serialize an optional object to string.
 *
 *  \code
 *    dmlc::optional<int> x;
 *    std::cout << x;  // None
 *    x = 0;
 *    std::cout << x;  // 0
 *  \endcode
 *
 *  \param os output stream
 *  \param t source optional<T> object
 *  \return output stream
 */
template<typename T>
std::ostream &operator<<(std::ostream &os, const optional<T> &t) {
  if (t) {
    os << *t;
  } else {
    os << "None";
  }
  return os;
}

/*! \brief parse a string object into optional<T>
 *
 *  \code
 *    dmlc::optional<int> x;
 *    std::string s1 = "1";
 *    std::istringstream is1(s1);
 *    s1 >> x;  // x == optional<int>(1)
 *
 *    std::string s2 = "None";
 *    std::istringstream is2(s2);
 *    s2 >> x;  // x == optional<int>()
 *  \endcode
 *
 *  \param is input stream
 *  \param t target optional<T> object
 *  \return input stream
 */
template<typename T>
std::istream &operator>>(std::istream &is, optional<T> &t) {
  char buf[4];
  std::streampos origin = is.tellg();
  is.read(buf, 4);
  if (is.fail() || buf[0] != 'N' || buf[1] != 'o' ||
      buf[2] != 'n' || buf[3] != 'e') {
    is.clear();
    is.seekg(origin);
    T x;
    is >> x;
    t = x;
    if (std::is_integral<T>::value && !is.eof() && is.peek() == 'L') is.get();
  } else {
    t = nullopt;
  }
  return is;
}
/*! \brief specialization of '>>' istream parsing for optional<bool>
 *
 * Permits use of generic parameter FieldEntry<DType> class to create
 * FieldEntry<optional<bool>> without explicit specialization.
 *
 *  \code
 *    dmlc::optional<bool> x;
 *    std::string s1 = "true";
 *    std::istringstream is1(s1);
 *    s1 >> x;  // x == optional<bool>(true)
 *
 *    std::string s2 = "None";
 *    std::istringstream is2(s2);
 *    s2 >> x;  // x == optional<bool>()
 *  \endcode
 *
 *  \param is input stream
 *  \param t target optional<bool> object
 *  \return input stream
 */
inline std::istream &operator>>(std::istream &is, optional<bool> &t) {
  // Discard initial whitespace
  while (isspace(is.peek()))
    is.get();
  // Extract chars that might be valid into a separate string, stopping
  // on whitespace or other non-alphanumerics such as ",)]".
  std::string s;
  while (isalnum(is.peek()))
    s.push_back(is.get());

  if (!is.fail()) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "1" || s == "true")
      t = true;
    else if (s == "0" || s == "false")
      t = false;
    else if (s == "none")
      t = nullopt;
    else
      is.setstate(std::ios::failbit);
  }

  return is;
}

/*! \brief description for optional int */
DMLC_DECLARE_TYPE_NAME(optional<int>, "int or None");
/*! \brief description for optional bool */
DMLC_DECLARE_TYPE_NAME(optional<bool>, "boolean or None");
/*! \brief description for optional float */
DMLC_DECLARE_TYPE_NAME(optional<float>, "float or None");
/*! \brief description for optional double */
DMLC_DECLARE_TYPE_NAME(optional<double>, "double or None");

}  // namespace dmlc

namespace std {
/*! \brief std hash function for optional */
template<typename T>
struct hash<dmlc::optional<T> > {
  /*!
   * \brief returns hash of the optional value.
   * \param val value.
   * \return hash code.
   */
  size_t operator()(const dmlc::optional<T>& val) const {
    std::hash<bool> hash_bool;
    size_t res = hash_bool(val.has_value());
    if (val.has_value()) {
      res = dmlc::HashCombine(res, val.value());
    }
    return res;
  }
};
}  // namespace std

#endif  // DMLC_OPTIONAL_H_
