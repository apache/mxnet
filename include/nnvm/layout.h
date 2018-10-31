/*!
 *  Copyright (c) 2018 by Contributors
 * \file nnvm/layout.h
 * \brief Layout expression.
 *        The layout is composed of upper cases, lower cases and numbers,
 *        where upper case indicates a (super-)dimension and
 *        the corresponding lower case with factor size indicates the split (sub-)dimension.
 *        For example, NCHW16c can describe a 5-D tensor of
 *        [batch_size, channel, height, width, channel_block].
 *        Here sub-dimension channel_block=16 is the split of super-dimension C (channel).
 */
#ifndef NNVM_LAYOUT_H_
#define NNVM_LAYOUT_H_

#include <dmlc/parameter.h>
#include <string>
#include <sstream>
#include <vector>
#include <utility>
#include <algorithm>

namespace nnvm {

class Layout {
 public:
  using LayoutDim = char;

  /*! \brief default constructor */
  Layout() : name_("__undef__") {} // NOLINT(*)

  /*!
   * \brief construct from a string.
   * \param layout input in layout convention:
   *        upper case indicates a dimension and
   *        the corresponding lower case with factor size
   *        indicates the split dimension.
   *        return undefined layout if "__undef__" is passed.
   */
  inline Layout(const std::string& layout) { // NOLINT(*)
    parse(layout);
  }
  /*!
   * \brief copy constructor from another layout
   * \param s the source layout
   */
  inline Layout(const Layout& s) { // NOLINT(*)
    this->parse(s.name_);
  }
  /*!
   * \brief move constructor from Layout
   * \param src the source layout
   */
  inline Layout(Layout&& src) { // NOLINT(*)
    this->swap(src);
  }
  /*!
   * \brief assignment from another layout.
   * \param src source layout
   * \return reference of self
   */
  inline Layout& operator=(const Layout& src) {
    this->parse(src.name_);
    return *this;
  }
  /*!
   * \brief assignment from rvalue of another layout.
   * \param src source layout
   * \return reference of self
   */
  inline Layout& operator=(Layout&& src) {
    Layout(std::move(src)).swap(*this); // NOLINT(*)
    return *this;
  }
  /*!
   * \brief assignment from string.
   * \param src source layout
   * \return reference of self
   */
  inline Layout& operator=(const std::string& src) {
    this->parse(src);
    return *this;
  }
  /*!
   * \return whether two layout equals
   * \param s the layout to compare against
   */
  inline bool operator==(const Layout& s) const {
    return name_ == s.name_;
  }
  /*!
   * \return whether two layout not equal
   * \param s the layout to compare against
   */
  inline bool operator!=(const Layout& s) const {
    return !(*this == s);
  }

  /*!
   * \brief Append the current layout by another.
   * @param other the layout to be appended
   * @return a new layout
   */
  inline Layout operator+(const Layout& other) const {
    if (!this->defined() && !other.defined()) {
      return Layout::Undef();
    } else if (!this->defined()) {
      return other;
    } else if (!other.defined()) {
      return *this;
    }
    return Layout(this->name_ + other.name_);
  }

  /*!
   * \brief Check whether a given dimension is a super-dimension.
   * \param dim input dimension
   * \return Whether a given dimension is a super-dimension.
   */
  static inline bool is_superdim(LayoutDim dim) {
    return dim >= 'A' && dim <= 'Z';
  }

  /*!
   * \brief Check whether a given dimension is a sub-dimension.
   * \param dim input dimension
   * \return Whether a given dimension is a sub-dimension.
   */
  static inline bool is_subdim(LayoutDim dim) {
    return dim >= 'a' && dim <= 'z';
  }

  /*!
   * \brief Convert a given dimension to super-dimension.
   * \param dim input dimension
   * \return The converted description.
   */
  static inline LayoutDim to_superdim(LayoutDim dim) {
    if (is_subdim(dim)) {
      return dim - 'a' + 'A';
    }
    return dim;
  }

  /*!
   * \brief Convert a given dimension to sub-dimension.
   * \param dim input dimension
   * \return The converted description.
   */
  static inline LayoutDim to_subdim(LayoutDim dim) {
    if (is_superdim(dim)) {
      return dim - 'A' + 'a';
    }
    return dim;
  }

  /*!
   * \brief Return an undefined layout.
   * \return a (global) undefined layout.
   */
  static inline const Layout& Undef() {
    static Layout undef;
    return undef;
  }

  /*!
   * \brief Swap current object with other
   * \param other another object to be swapped.
   */
  inline void swap(Layout& other) {  // NOLINT(*)
    std::swap(name_, other.name_);
    std::swap(superdim_pos_, other.superdim_pos_);
    std::swap(subdim_pos_, other.subdim_pos_);
    std::swap(subdim_size_, other.subdim_size_);
    std::swap(layout_simplified_, other.layout_simplified_);
  }

  /*!
   * \brief Two layouts are convertible only if
   *        they have same set of super-dimensions.
   *        e.g., NCHW, NCHW16c, NHWC are convertible between each other,
   *        but NCHW, CHW, OIHW are not.
   * \param dst the target layout
   * \return Whether can be converted to dst layout.
   */
  inline bool convertible(const Layout &dst) const {
    if (!this->defined() || !dst.defined()) return false;
    for (size_t i = 0; i < kUniqueDim; ++i) {
      if ((superdim_pos_[i] >= 0 && dst.superdim_pos_[i] < 0) ||
          (superdim_pos_[i] < 0 && dst.superdim_pos_[i] >= 0)) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Returns a sublayout which is the portion of the object
   *        that starts at dimension \p pos and spans \p len dimensions
   *        (or until the end of the layout, whichever comes first).
   * \param pos The start position.
   * \param len The length of the sub-layout.
   * \return A newly constructed Layout object.
   */
  inline Layout sublayout(size_t pos, size_t len) const {
    if (pos > ndim()) return Layout::Undef();
    if (pos + len > ndim()) len = ndim() - pos;
    if (len == 0) return Layout::Undef();
    std::ostringstream new_layout;
    for (size_t i = pos; i < pos + len; ++i) {
      if (is_subdim(layout_simplified_[i])) {
        auto block_size = this->subsizeof(layout_simplified_[i]);
        CHECK_GT(block_size, 0);
        new_layout << block_size;
      }
      new_layout << layout_simplified_[i];
    }
    return Layout(new_layout.str());
  }

  /*! \return A newly constructed reversed Layout object. */
  inline Layout reverse() const {
    if (!this->defined()) return Layout::Undef();
    std::ostringstream new_layout;
    for (int64_t i = this->ndim() - 1; i >= 0; --i) {
      if (is_subdim(layout_simplified_[i])) {
        auto block_size = this->subsizeof(layout_simplified_[i]);
        CHECK_GT(block_size, 0);
        new_layout << block_size;
      }
      new_layout << layout_simplified_[i];
    }
    return Layout(new_layout.str());
  }

  /*!
   * \brief Split \p dim by \p size and put the sub-dimension to position \p target_pos.
   * \param dim The source dimension to be split. It must be a super-dimension.
   * \param target_pos The target position of the newly split sub-dimension.
   * \param size size of the sub-dimension.
   * \return A newly constructed Layout object.
   */
  inline Layout split(LayoutDim dim, size_t target_pos, uint32_t size) const {
    CHECK(target_pos <= this->ndim()) << "Invalid split position "
                                      << target_pos << " for layout " << name_;
    CHECK(is_superdim(dim)) << "Cannot split a sub-dimension " << dim;
    CHECK(this->contains(dim)) << "Axis " << dim << " does not exist in " << name_;
    CHECK(!this->contains(to_subdim(dim))) << "Dimension " << dim
                                           << " has already been split in "
                                           << name_;
    CHECK(size > 0) << "Invalid split size " << size;
    std::ostringstream new_layout;
    for (size_t i = 0; i <= this->ndim(); ++i) {
      if (i == target_pos) {
        new_layout << size << Layout::to_subdim(dim);
      }
      if (i == this->ndim()) break;
      new_layout << this->at(i);
    }
    Layout x(new_layout.str());
    return x;
  }

  using iterator = std::vector<LayoutDim>::const_iterator;
  using reverse_iterator = std::vector<LayoutDim>::const_reverse_iterator;

  /*! \return begin iterator */
  inline iterator begin() const {
    return layout_simplified_.begin();
  }
  /*! \return end iterator */
  inline iterator end() const {
    return layout_simplified_.end();
  }
  /*! \return rbegin iterator */
  inline reverse_iterator rbegin() const {
    return layout_simplified_.rbegin();
  }
  /*! \return rend iterator */
  inline reverse_iterator rend() const {
    return layout_simplified_.rend();
  }

  /*! \return number of dimensions */
  inline size_t ndim() const {
    return layout_simplified_.size();
  }

  /*!
   * \brief The description of the \p i-th dimension.
   *        If it is a sub-dimension, the size will be returned as well,
   *        e.g., 16c. Otherwise a single character is returned, e.g., C.
   * \param i The position
   * \return the description of the dimension.
   */
  inline std::string at(size_t i) const {
    CHECK_LT(i, this->ndim()) << "position " << i
                              << " exceeds ndim=" << this->ndim();
    std::ostringstream repr;
    if (is_subdim(layout_simplified_[i])) {
      auto factor = subsizeof(layout_simplified_[i]);
      CHECK_GT(factor, 0);
      repr << factor;
    }
    repr << layout_simplified_[i];
    return repr.str();
  }

  /*!
   * \brief return the index of the input dimension.
   *        If it is not found in the layout or the layout is undefined,
   *        return -1.
   * \param dim the input dimension.
   * \return the index or -1 if not found.
   */
  inline int32_t indexof(LayoutDim dim) const {
    if (!this->defined()) return -1;
    else if (is_superdim(dim)) return superdim_pos_[dim - 'A'];
    else if (is_subdim(dim)) return subdim_pos_[dim - 'a'];
    return -1;
  }

  /*!
   * \param dim the input super-dimension or sub-dimension.
   * \return the size of the sub-dimension of \p dim (if \p dim is a super-dimension),
   *         or the size of \p dim itself (if \p dim is a sub-dimension).
   *         Return -1 if \p dim is not in the layout or the layout is undefined.
   */
  inline int64_t subsizeof(LayoutDim dim) const {
    CHECK(is_superdim(dim) || is_subdim(dim)) << "Invalid dim " << dim;
    if (!this->defined() || !this->contains(to_subdim(dim))) {
      return -1;
    }
    int idx = to_subdim(dim) - 'a';
    return subdim_size_[idx];
  }

  /*!
   * \brief Whether the layout contains a dimension.
   * \param dim dimension to be checked.
   * \return Whether the layout contains the dimension.
   */
  inline bool contains(LayoutDim dim) const {
    if (is_superdim(dim)) {
      return superdim_pos_[dim-'A'] >= 0;
    } else if (is_subdim(dim)) {
      return subdim_pos_[dim-'a'] >= 0;
    }
    return false;
  }

  inline LayoutDim operator[](size_t i) const {
    return layout_simplified_[i];
  }

  /*! \return whether the layout is defined */
  inline bool defined() const {
    return name_ != "__undef__";
  }

  /*! \return the string description of the layout */
  inline const std::string& name() const {
    return name_;
  }

  /*!
   * \brief Write layout in JSON format.
   * \param writer JSONWriter
   */
  inline void Save(dmlc::JSONWriter* writer) const {
    writer->Write(name_);
  }

  /*!
   * \brief Load layout from JSON.
   * \param reader JSONReader
   */
  inline void Load(dmlc::JSONReader* reader) {
    std::string tmp;
    reader->Read(&tmp);
    this->parse(tmp);
  }

  /*!
   * \brief allow output string of layout to ostream
   * \param os the output stream
   * \param l the layout
   * \return the ostream
   */
  friend std::ostream& operator<<(std::ostream& os, const Layout& l) {
    os << l.name_;
    return os;
  }

 private:
  static const uint32_t kUniqueDim = 26;

  std::string name_;
  int32_t superdim_pos_[kUniqueDim];
  int32_t subdim_pos_[kUniqueDim];
  int64_t subdim_size_[kUniqueDim];
  std::vector<LayoutDim> layout_simplified_;

  void parse(const std::string& layout) {
    name_ = layout;
    std::fill_n(superdim_pos_, kUniqueDim, -1);
    std::fill_n(subdim_pos_, kUniqueDim, -1);
    std::fill_n(subdim_size_, kUniqueDim, -1);
    layout_simplified_.clear();

    if (layout == "__undef__") return;

    int32_t factor = 0;
    uint32_t curr = 0;
    for (size_t i = 0; i < layout.size(); ++i) {
      const LayoutDim c = layout.at(i);
      if (is_superdim(c)) {
        int pos = c - 'A';
        CHECK_EQ(factor, 0) << "Invalid layout " << layout
                            << ": invalid factor size " << factor
                            << " before dimension " << c;
        CHECK_EQ(superdim_pos_[pos], -1) << "Invalid layout " << layout
                                           << ": duplicate dimension " << c;
        superdim_pos_[pos] = curr++;
        layout_simplified_.push_back(c);
      } else if (is_subdim(c)) {
        int pos = c - 'a';
        CHECK_GT(factor, 0) << "Invalid layout " << layout << ": invalid factor size "
                            << factor << " for dimension " << c;
        CHECK_EQ(subdim_pos_[pos], -1) << "Invalid layout " << layout
                                           << ": duplicate dimension " << c;
        CHECK_EQ(subdim_size_[pos], -1) << "Invalid layout " << layout
                                         << ": duplicate dimension " << c;
        subdim_pos_[pos] = curr++;
        subdim_size_[pos] = factor;
        layout_simplified_.push_back(c);
        factor = 0;
      } else if (c >= '0' && c <= '9') {
        CHECK(factor >= 0) << "Invalid layout " << layout << ": _ is adjacent to a number.";
        factor = factor * 10 + c - '0';
      } else {
        LOG(FATAL) << "Invalid layout " << layout;
      }
    }
    CHECK(!layout_simplified_.empty()) << "Invalid layout " << layout;
    for (LayoutDim dim : layout_simplified_) {
      CHECK(is_superdim(dim) || superdim_pos_[dim-'a'] >= 0)
        << "Invalid layout " << layout << ": missing axis "
        << static_cast<char>(dim - 'a' + 'A');
    }
  }
};

}  // namespace nnvm

#endif  // NNVM_LAYOUT_H_
