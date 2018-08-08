//===-- ShapedArray.swift -------------------------------------*- swift -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

import Swift
import CTensorFlow

//===----------------------------------------------------------------------===//
// ShapedArrayProtocol, the protocol unifying ShapedArray and ShapedArraySlice.
//===----------------------------------------------------------------------===//

public protocol _ShapedArrayProtocol
  : RandomAccessCollection, MutableCollection {
  associatedtype Scalar
  associatedtype ScalarCollection : RandomAccessCollection, MutableCollection
      where ScalarCollection.Element == Scalar

  /// The number of dimensions of the array.
  var rank: Int { get }
  /// The dimensions of the array.
  var shape: [Int] { get }
  /// The scalars in row-major order.
  var scalars: ScalarCollection { get set }
  /// The total number of scalars in the array.
  var scalarCount: Int { get }

  /// Creates an array with the specified shape and contiguous scalars in
  /// row-major order.
  /// - Precondition: The number of scalars must equal the product of the
  ///   dimensions of the shape.
  init(shape: [Int], scalars: [Scalar])

  /// Creates an array with the specified shape and sequence of scalars in
  /// row-major order.
  /// - Precondition: The number of scalars must equal the product of the
  ///   dimensions of the shape.
  init<S : Sequence>(shape: [Int], scalars: S) where S.Element == Scalar

  /// Calls a closure with a pointer to the array’s contiguous storage.
  /// - Parameter body: A closure with an `UnsafeBufferPointer` parameter that
  ///   points to the contiguous storage for the array. If no such storage
  ///   exists, it is created. If body has a return value, that value is also
  ///   used as the return value for the `withUnsafeBufferPointer(_:)` method.
  ///   The pointer argument is valid only for the duration of the method’s
  ///   execution.
  func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> R
  ) rethrows -> R

  /// Calls the given closure with a pointer to the array’s mutable contiguous
  /// storage.
  /// - Parameter body: A closure with an `UnsafeMutableBufferPointer` parameter
  ///   that points to the contiguous storage for the array. If no such storage
  ///   exists, it is created. If body has a return value, that value is also
  ///   used as the return value for the `withUnsafeMutableBufferPointer(_:)`
  ///   method. The pointer argument is valid only for the duration of the
  ///   method’s execution.
  mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> R
  ) rethrows -> R
}

public extension _ShapedArrayProtocol {
  /// Returns `true` if the array has rank 0.
  var isScalar: Bool {
    return rank == 0
  }

  /// Returns the single scalar element if the array has rank 0 and `nil`
  /// otherwise.
  var scalar: Scalar? {
    guard rank == 0 else { return nil }
    return scalars.first
  }
}

public extension _ShapedArrayProtocol where Scalar : Equatable {
  static func == <Other>(lhs: Self, rhs: Other) -> Bool
    where Other : _ShapedArrayProtocol, Scalar == Other.Scalar {
    return lhs.shape == rhs.shape && lhs.scalars.elementsEqual(rhs.scalars)
  }
}

public extension _ShapedArrayProtocol {
  /// Returns the number of element arrays in an array (equivalent to the first
  /// dimension).
  /// - Note: `count` is distinct from `scalarCount`, which represents the total
  ///   number of scalars.
  var count: Int {
    return shape.first ?? 0
  }
}

internal extension _ShapedArrayProtocol {
  /// Returns the scalar count for an element of the array.
  var scalarCountPerElement: Int {
    return shape.isEmpty ? 0 : shape.dropFirst().reduce(1, *)
  }

  /// Returns the scalar index corresponding to an index in the leading
  /// dimension of the array.
  func scalarIndex(fromIndex index: Int) -> Int {
    return scalarCountPerElement * index
  }

  /// Returns the range of scalars corresponding to a range in the leading
  /// dimension of the array.
  func scalarSubrange(
    from arraySubrange: Range<Int>
  ) -> Range<Int> {
    return scalarIndex(fromIndex: arraySubrange.lowerBound)
      ..< scalarIndex(fromIndex: arraySubrange.upperBound)
  }
}

/// Common public protocol implementations
fileprivate extension _ShapedArrayProtocol
  where Element : _ShapedArrayProtocol {
  var _description: String {
    if let scalar = scalar {
      return String(describing: scalar)
    }
    return "[\( map({"\($0)"}).joined(separator: ", ") )]"
  }
}

fileprivate extension _ShapedArrayProtocol where Scalar : Equatable {
  func _isEqual(to other: Self) -> Bool {
    return shape == other.shape && scalars.elementsEqual(other.scalars)
  }
}

//===----------------------------------------------------------------------===//
// ShapedArray
//===----------------------------------------------------------------------===//

/// `ShapedArray` is a multi-dimensional array. It has a shape, which has type
/// `[Int]` and defines the array dimensions, and uses an array of `Scalar`
/// internally as storage.
@_fixed_layout
public struct ShapedArray<Scalar> : _ShapedArrayProtocol {
  /// Array of scalars
  public var scalars: [Scalar] {
    willSet {
      precondition(newValue.count == scalarCount, "Scalar count mismatch.")
    }
  }

  /// The dimensions of the array.
  public private(set) var shape: [Int]

  /// Creates a `ShapedArray` from an array and a shape.
  public init(shape: [Int], scalars: [Scalar]) {
    precondition(scalars.count == shape.reduce(1, *),
      "The scalar count of the buffer does not match the shape.")
    self.scalars = scalars
    self.shape = shape
    debugLog("Done initializing ShapedArray from [Scalar].")
  }
}

internal extension ShapedArray where Scalar : AccelerableByTensorFlow {
  @usableFromInline
  init(cTensor: CTensor) {
    print("Initialize from cTensor")
    // Including \(Scalar.self) into the message would cause non-deterministic
    // crashes.
    debugLog("Initializing ShapedArray from CTensor.")
    self.shape = (0..<TF_NumDims(cTensor)).map { Int(TF_Dim(cTensor, $0)) }
    if _RuntimeConfig.printsDebugLog {
      // Without this local variable, passing the string directly into
      // debugLog() would not work, because 'self' is captured by the auto
      // closure param in debugLog().
      let shapeStr = "The shape is \(shape)."
      debugLog(shapeStr)
    }
    let count = shape.reduce(1, *)
    let cTensorPtr = TF_TensorData(cTensor)!
    let cTensorSize = TF_TensorByteSize(cTensor)
    let dtype = TF_TensorType(cTensor)
    if dtype == TF_STRING {
      // decode string tensors and store
      var stringArray: [String] = []
      let offsetSize = count * MemoryLayout<UInt64>.stride
      let strStart = cTensorPtr.advanced(by: offsetSize)
                               .assumingMemoryBound(to: Int8.self)
      let offsetBuf = cTensorPtr.bindMemory(to: UInt64.self, capacity: count)
      let cTensorEnd = cTensorPtr + cTensorSize
      var nextStrStart = strStart
      var cString: UnsafePointer<Int8>?
      var cStringLen: Int = 0
      let status = TF_NewStatus()
      for i in 0..<count {
        precondition(nextStrStart == strStart.advanced(by: Int(offsetBuf[i])),
                     "Incorrect pointer to read")
        let bytesRead = TF_StringDecode(nextStrStart,
                                        UnsafeRawPointer(nextStrStart)
                                          .distance(to: cTensorEnd),
                                        &cString, &cStringLen, status)
        print("bytesRead \(bytesRead)")
        checkOk(status)
        nextStrStart = nextStrStart.advanced(by: bytesRead)
        stringArray.append(String(cString: cString!))
      }
      TF_DeleteStatus(status)
      self.scalars = stringArray as! [Scalar]
    }
    else {
      let dummyPointer = UnsafeMutablePointer<Scalar>.allocate(capacity: 1)
      self.scalars = Array(repeating: dummyPointer.move(), count: count)
      dummyPointer.deallocate()
      // perform memcpy for non-String scalars
      let tensorData = cTensorPtr.assumingMemoryBound(to: Scalar.self)
      self.scalars.withUnsafeMutableBufferPointer { [count] bufPtr in
        bufPtr.baseAddress?.initialize(from: tensorData, count: count)
      }
    }
    debugLog("Done initializing ShapedArray from CTensor.")
  }
}

public extension ShapedArray {
  var rank: Int {
    return shape.count
  }

  var scalarCount: Int {
    return scalars.count
  }

  init(_ other: ShapedArray) {
    debugLog("Initializing from another ShapedArray.")
    self.init(shape: other.shape, scalars: other.scalars)
  }

  init<S : Sequence>(shape: [Int], scalars: S) where S.Element == Scalar {
    let scalarCount = shape.reduce(1, *)
    let scalars = Array(scalars)
    // If the sequence has fewer elements than the shape needs, this is a
    // precondition failure.
    precondition(scalars.count == scalarCount,
                   "The sequence has fewer elements than needed by the shape.")
    self.init(shape: shape, scalars: scalars)
  }

  /// Creates a `ShapedArray` from a scalar value.
  init(_ scalar: Scalar) {
    let element = [scalar]
    self.init(shape: [], scalars: element)
  }

  /// Creates a `ShapedArray` with the specified shape and a single, repeated
  /// value.
  /// - Parameters:
  ///   - shape: The dimensions of the array.
  ///   - repeatedValue: The scalar value to repeat.
  init(shape: [Int], repeating repeatedValue: Scalar) {
    let scalarCount = shape.reduce(1, *)
    let scalars = Array(repeating: repeatedValue, count: scalarCount)
    self.init(shape: shape, scalars: scalars)
  }
}

extension ShapedArray : RandomAccessCollection, MutableCollection {
  public typealias Index = Int
  public typealias Element = ShapedArraySlice<Scalar>
  public typealias SubSequence = ShapedArraySlice<Scalar>

  public var indices: Range<Int> {
    return 0..<count
  }

  public var startIndex: Int {
    return 0
  }

  public var endIndex: Int {
    return count
  }

  /// Access the element array specified by an index in the leading dimension.
  /// - Parameter index: Index of the element array.
  public subscript(index: Int) -> Element {
    get {
      precondition(!isScalar,
                   "Scalar has no elements and cannot be subscripted.")
      precondition(index < endIndex, "ShapedArray index is out of range")
      precondition(index >= startIndex,
                   "Negative ShapedArray index is out of range")
      return ShapedArraySlice(base: self, baseIndices: [index])
    }
    set {
      precondition(!isScalar,
                   "Scalar has no elements and cannot be subscripted.")
      precondition(index < endIndex, "ShapedArray index is out of range")
      precondition(index >= startIndex,
                   "Negative ShapedArray index is out of range")
      precondition(shape.dropFirst().elementsEqual(newValue.shape),
                   "Element shape mismatch")
      let scalarIndex = self.scalarIndex(fromIndex: index)
      withUnsafeMutableBufferPointer { destBuffPtr in
        let ptr = destBuffPtr.baseAddress!.advanced(by: scalarIndex)
        newValue.withUnsafeBufferPointer { srcBuffPtr in
          ptr.initialize(from: srcBuffPtr.baseAddress!, count: srcBuffPtr.count)
        }
      }
    }
  }

  /// Access the subarray specified by a contiguous range of indices.
  /// - Parameter bounds: Contiguous range of indices.
  public subscript(bounds: Range<Int>) -> SubSequence {
    get {
      precondition(!isScalar,
                   "Scalar has no elements and cannot be subscripted.")
      precondition(
        indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
        "ShapedArray indices are out of range")
      return ShapedArraySlice(base: self, bounds: bounds)
    }
    set {
      precondition(!isScalar,
                   "Scalar has no elements and cannot be subscripted.")
      precondition(
        indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
        "ShapedArray indices are out of range")
      let subArrayShape = [bounds.count] + shape.dropFirst()
      precondition(subArrayShape == newValue.shape,
                   "Subarray shape mismatch.")
      let scalarIndex = self.scalarIndex(fromIndex: bounds.lowerBound)
      withUnsafeMutableBufferPointer { destBuffPtr in
        let ptr = destBuffPtr.baseAddress!.advanced(by: scalarIndex)
        newValue.withUnsafeBufferPointer { srcBuffPtr in
          ptr.initialize(from: srcBuffPtr.baseAddress!, count: srcBuffPtr.count)
        }
      }
    }
  }
}

public extension ShapedArray {
  func withUnsafeBufferPointer<Result>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> Result
  ) rethrows -> Result {
    return try scalars.withUnsafeBufferPointer { ptr in
      try body(ptr)
    }
  }

  mutating func withUnsafeMutableBufferPointer<Result>(
    _ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> Result
  ) rethrows -> Result {
    return try scalars.withUnsafeMutableBufferPointer { ptr in
      try body(&ptr)
    }
  }
}

/// Tensor conversion
extension ShapedArray where Scalar : AccelerableByTensorFlow {
  var byteCount: Int {
    return MemoryLayout<Scalar>.stride * scalarCount
  }

  @usableFromInline
  func makeTensorHandle() -> TensorHandle<Scalar> {
    // This initializer is designed to optimize conversion from TF-allocated
    // `ShapedArray` instances.
    precondition(rank <= Int32.max, """
      Conversion to TensorHandle is undefined when rank exceeds Int32.max.
      """)
    precondition(shape.forAll { $0 <= Int32.max }, """
      Conversion to TensorHandle is undefined when shape dimensions exceed \
      Int32.max.
      """)
    return TensorHandle<Scalar>(
      shape: shape.map(Int32.init),
      scalarsInitializer: { addr in
        // FIXME
        addr.initialize(from: scalars, count: scalarCount)
      }
    )
  }
}

/// Tensor conversion
public extension Tensor where Scalar : AccelerableByTensorFlow {
  init(_ array: ShapedArray<Scalar>) {
    self.init(handle: array.makeTensorHandle())
  }
}

/// Array literal conversion
extension ShapedArray : ExpressibleByArrayLiteral
  where Scalar : AccelerableByTensorFlow {
  public typealias ArrayLiteralElement = TensorElementLiteral<Scalar>
  @inlinable @inline(__always)
  public init(arrayLiteral elements: TensorElementLiteral<Scalar>...) {
    self = Tensor<Scalar>(tensorElementLiterals: elements).array
  }
}

/// Equatable conformance
extension ShapedArray : Equatable where Scalar : Equatable {
  public static func == (lhs: ShapedArray, rhs: ShapedArray) -> Bool {
    return lhs._isEqual(to: rhs)
  }
}

/// String conversion
extension ShapedArray : CustomStringConvertible {
  /// A textual representation of this `ShapedArray`.
  public var description: String {
    return _description
  }
}

/// Xcode Playground display conversion.
extension ShapedArray : CustomPlaygroundDisplayConvertible {
  public var playgroundDescription: Any {
    return description
  }
}

/// Mirror representation, used by debugger/REPL.
extension ShapedArray : CustomReflectable {
  public var customMirror: Mirror {
    return Mirror(self, children: [], displayStyle: .struct)
  }
}

//===----------------------------------------------------------------------===//
// ShapedArraySlice
//===----------------------------------------------------------------------===//

/// A contiguous slice of a `ShapedArray` or `ShapedArraySlice` instance.
///
/// `ShapedArraySlice` enables fast, efficient operations on contiguous slices
/// of `ShapedArray` instances. `ShapedArraySlice` instances do not have their
/// own storage. Instead, they provides a view onto the storage of their base
/// `ShapedArray`. `ShapedArraySlice` can represent two different kinds of
/// slices: element arrays and subarrays.
///
/// Element arrays are subdimensional elements of a `ShapedArray`: their rank
/// is one less than that of their base. Element array slices are obtained by
/// indexing a `ShapedArray` instance with a singular `Int32` index.
///
/// For example:
///
///     let matrix = ShapedArray(shape: [2, 2], scalars: [0, 1, 2, 3])
///     // `matrix` represents [[0, 1], [2, 3]].
///
///     let element = matrix[0]
///     // `element` is a `ShapedArraySlice` with shape [2]. It is an element
///     // array, specifically the first element in `matrix`: [0, 1].
///
///     matrix[1] = ShapedArraySlice(shape: [2], scalars: [4, 8])
///     // The second element in `matrix` has been mutated.
///     // `matrix` now represents [[0, 1, 4, 8]].
///
/// Subarrays are a contiguous range of the elements in a `ShapedArray`.
/// The rank of a subarray is the same as that of its base, but its leading
/// dimension is the count of the slice range. Subarray slices are obtained by
/// indexing a `ShapedArray` with a `Range<Int32>` that represents a range of
/// elements (in the leading dimension). Methods like `prefix(:)` and
/// `suffix(:)` that internally index with a range also produce subarray.
///
/// For example:
///
///     let zeros = ShapedArray(shape: [3, 2], repeating: 0)
///     var matrix = ShapedArray(shape: [3, 2], scalars: Array(0..<6))
///     // `zeros` represents [[0, 0], [0, 0], [0, 0]].
///     // `matrix` represents [[0, 1], [2, 3], [4, 5]].
///
///     let subarray = matrix.prefix(2)
///     // `subarray` is a `ShapedArraySlice` with shape [2, 2]. It is a slice
///     // of the first 2 elements in `matrix` and represents [[0, 1], [2, 3]].
///
///     matrix[0..<2] = zeros.prefix(2)
///     // The first 2 elements in `matrix` have been mutated.
///     // `matrix` now represents [[0, 0], [0, 0], [4, 5]].

@_fixed_layout
public struct ShapedArraySlice<Scalar> : _ShapedArrayProtocol {
  /// The underlying `ShapedArray` of the slice.
  @usableFromInline internal var base: ShapedArray<Scalar>
  /// The subdimensional indices of a slice.
  @usableFromInline internal var baseIndices: [Int]
  /// The subarray bounds of a slice.
  @usableFromInline internal var bounds: Range<Int>?

  /// Creates a `ShapedArraySlice` from a base `ShapedArray`, with the specified
  /// subdimensional indices and subarray bounds.
  @inlinable
  internal init(
    base: ShapedArray<Scalar>,
    baseIndices indices: [Int] = [],
    bounds: Range<Int>? = nil
  ) {
    precondition(indices.count <= base.rank,
                 "Number of base indices exceeds base rank")
    precondition(zip(base.shape, indices).forAll { $1 >= 0 && $1 < $0 },
                 "Base indices are out of range")
    self.base = base
    self.baseIndices = indices
    self.bounds = bounds
  }
}

public extension ShapedArraySlice {
  /// Indexing depth of this slice, i.e. the difference in rank between the base
  /// and the slice.
  internal var indexingDepth: Int {
    return baseIndices.count
  }

  var rank: Int {
    return base.rank - indexingDepth
  }

  var shape: [Int] {
    if let bounds = bounds {
      return [bounds.count] + Array(base.shape.dropFirst(indexingDepth + 1))
    }
    return Array(base.shape.dropFirst(indexingDepth))
  }

  var scalarCount: Int {
    return shape.reduce(1, *)
  }
  
  var scalars: ArraySlice<Scalar> {
    get {
      return base.scalars[scalarRange]
    }
    set {
      base.scalars[scalarRange] = newValue
    }
  }
}

/// Slice initializers
public extension ShapedArraySlice {
  init(shape: [Int], scalars: [Scalar]) {
    self.init(base: ShapedArray(shape: shape, scalars: scalars))
  }

  init<S : Sequence>(shape: [Int], scalars: S) where S.Element == Scalar {
    self.init(base: ShapedArray(shape: shape, scalars: scalars))
  }

  /// Creates a `ShapedArraySlice` from a scalar value.
  init(_ scalar: Scalar) {
    self.init(base: ShapedArray(scalar))
  }

  /// Creates a `ShapedArraySlice` with the specified shape and a single,
  /// repeated value.
  /// - Parameters:
  ///   - shape: The dimensions of the array.
  ///   - repeatedValue: The scalar value to repeat.
  init(shape: [Int], repeating repeatedValue: Scalar) {
    self.init(base: ShapedArray(shape: shape, repeating: repeatedValue))
  }
}

internal extension ShapedArraySlice {
  /// The range of scalars from the base `ShapedArray` represented by a
  /// `ShapedArraySlice`.
  var scalarRange: Range<Int> {
    let trimmedShape = base.shape.dropFirst()
    var (start, end) = baseIndices.enumerated()
      .reduce((0, base.scalarCount)) { (acc, next) in
      let stride = trimmedShape.dropFirst(next.offset).reduce(1, *)
      if next.offset == indexingDepth - 1 {
        let temp = acc.0 + next.element * stride
        return (temp, temp + stride)
      }
      return (acc.0 + next.element * stride, acc.1)
    }
    if let bounds = bounds {
      let stride = trimmedShape.dropFirst(indexingDepth).reduce(1, *)
      let oldStart = start
      start = start + bounds.startIndex * stride
      end = oldStart + bounds.endIndex * stride
    }
    return start..<end
  }
}

public extension ShapedArraySlice {
  func withUnsafeBufferPointer<Result>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> Result
  ) rethrows -> Result {
    return try base.withUnsafeBufferPointer { baseBuffPtr in
      let basePtr = baseBuffPtr.baseAddress!
      let ptr = UnsafeBufferPointer(
        start: basePtr.advanced(by: scalarRange.startIndex),
        count: scalarRange.count
      )
      return try body(ptr)
    }
  }

  mutating func withUnsafeMutableBufferPointer<Result>(
    _ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> Result
  ) rethrows -> Result {
    // NOTE: Copying `scalarRange` to a local variable here is necessary for
    // exclusive access.
    let scalarRange = self.scalarRange
    return try base.withUnsafeMutableBufferPointer { baseBuffPtr in
      let basePtr = baseBuffPtr.baseAddress!
      var ptr = UnsafeMutableBufferPointer(
        start: basePtr.advanced(by: scalarRange.startIndex),
        count: scalarRange.count
      )
      return try body(&ptr)
    }
  }
}

extension ShapedArraySlice : RandomAccessCollection, MutableCollection {
  public typealias Index = Int
  public typealias Element = ShapedArraySlice
  public typealias SubSequence = ShapedArraySlice

  public var indices: Range<Int> {
    if let bounds = bounds {
      return bounds
    } else if indexingDepth < base.rank {
      return 0..<base.shape[indexingDepth]
    }
    return 0..<0
  }

  public var startIndex: Int {
    return indices.startIndex
  }

  public var endIndex: Int {
    return indices.endIndex
  }

  /// Access the element array specified by an index in the leading dimension.
  /// - Parameter index: Index of the element array.
  public subscript(index: Int) -> Element {
    get {
      precondition(!isScalar,
                   "Scalar has no elements and cannot be subscripted.")
      precondition(index < endIndex, "ShapedArraySlice index is out of range")
      precondition(index >= startIndex,
                   "ShapeArraySlice index is out of range (before startIndex)")
      return ShapedArraySlice(base: base,
                              baseIndices: baseIndices + [index],
                              bounds: bounds)
    }
    set {
      precondition(!isScalar,
                   "Scalar has no elements and cannot be subscripted.")
      precondition(index < endIndex, "ShapedArraySlice index is out of range")
      precondition(index >= startIndex,
                   "ShapeArraySlice index is out of range (before startIndex)")
      precondition(shape.dropFirst().elementsEqual(newValue.shape),
                   "Element shape mismatch")
      let scalarIndex = self.scalarIndex(fromIndex: index)
      withUnsafeMutableBufferPointer { destBuffPtr in
        let ptr = destBuffPtr.baseAddress!.advanced(by: scalarIndex)
        newValue.withUnsafeBufferPointer { srcBuffPtr in
          ptr.initialize(from: srcBuffPtr.baseAddress!, count: srcBuffPtr.count)
        }
      }
    }
  }

  /// Access the subarray specified by a contiguous range of indices.
  /// - Parameter bounds: Contiguous range of indices.
  public subscript(bounds: Range<Int>) -> SubSequence {
    get {
      precondition(!isScalar,
                   "Scalar has no elements and cannot be subscripted")
      precondition(
        indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
        "ShapedArraySlice indices are out of range")
      return ShapedArraySlice(base: base,
                              baseIndices: baseIndices,
                              bounds: bounds)
    }
    set {
      precondition(!isScalar,
                   "Scalar has no elements and cannot be subscripted")
      precondition(
        indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
        "ShapedArraySlice indices are out of range")
      let subArrayShape = [bounds.count] + shape.dropFirst()
      precondition(subArrayShape == newValue.shape, "Subarray shape mismatch")
      let scalarIndex = self.scalarIndex(fromIndex: bounds.lowerBound)
      withUnsafeMutableBufferPointer { destBuffPtr in
        let ptr = destBuffPtr.baseAddress!.advanced(by: scalarIndex)
        newValue.withUnsafeBufferPointer { srcBuffPtr in
          ptr.initialize(from: srcBuffPtr.baseAddress!, count: srcBuffPtr.count)
        }
      }
    }
  }
}

/// Tensor conversion
public extension ShapedArraySlice where Scalar : AccelerableByTensorFlow {
  init(_ tensor: Tensor<Scalar>) {
    self.init(base: tensor.array)
  }
}

/// Array literal conversion
extension ShapedArraySlice : ExpressibleByArrayLiteral
  where Scalar : AccelerableByTensorFlow {
  public typealias ArrayLiteralElement = TensorElementLiteral<Scalar>
  @inlinable @inline(__always)
  public init(arrayLiteral elements: TensorElementLiteral<Scalar>...) {
    self.init(base: Tensor(tensorElementLiterals: elements).array)
  }
}

/// Equatable conformance
extension ShapedArraySlice : Equatable where Scalar : Equatable {
  public static func == (lhs: ShapedArraySlice, rhs: ShapedArraySlice) -> Bool {
    return lhs._isEqual(to: rhs)
  }
}

/// String conversion
extension ShapedArraySlice : CustomStringConvertible {
  /// A textual representation of this `ShapedArraySlice`.
  public var description: String {
    return _description
  }
}

/// Xcode Playground display conversion
extension ShapedArraySlice : CustomPlaygroundDisplayConvertible {
  public var playgroundDescription: Any {
    return description
  }
}

/// Mirror representation, used by debugger/REPL.
extension ShapedArraySlice : CustomReflectable {
  public var customMirror: Mirror {
    return Mirror(self, children: [], displayStyle: .struct)
  }
}
