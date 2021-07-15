package org.apache.mxnet.ndarray;

import org.apache.mxnet.ndarray.types.Shape;

import java.util.Arrays;

/** This class contains various methods for manipulating MxNDArrays. */
public final class MxNDArrays {

    private MxNDArrays() {}

    private static void checkInputs(MxNDArray[] arrays) {
        if (arrays == null || arrays.length < 2) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        }
        if (arrays.length > 2
                && Arrays.stream(arrays).skip(1).anyMatch(array -> !arrays[0].shapeEquals(array))) {
            throw new IllegalArgumentException("The shape of all inputs must be the same");
        }
    }

    ////////////////////////////////////////
    // Operations: Element Comparison
    ////////////////////////////////////////

    /**
     * Returns {@code true} if all elements in {@link MxNDArray} a are equal to {@link MxNDArray} b.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.ones(new Shape(3));
     * jshell&gt; MxNDArrays.contentEquals(array, 1); // return true instead of boolean MxNDArray
     * true
     * </pre>
     *
     * @param a the {@link MxNDArray} to compare
     * @param n the number to compare
     * @return the boolean result
     */
    public static boolean contentEquals(MxNDArray a, Number n) {
        if (a == null) {
            return false;
        }
        return a.contentEquals(n);
    }

    /**
     * Returns {@code true} if all elements in {@link MxNDArray} a are equal to {@link MxNDArray} b.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.arange(6f).reshape(2, 3);
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 5f}, new Shape(2, 3));
     * jshell&gt; MxNDArrays.contentEquals(array1, array2); // return true instead of boolean MxNDArray
     * true
     * </pre>
     *
     * @param a the {@link MxNDArray} to compare
     * @param b the {@link MxNDArray} to compare
     * @return the boolean result
     */
    public static boolean contentEquals(MxNDArray a, MxNDArray b) {
        return a.contentEquals(b);
    }

    /**
     * Checks 2 {@link MxNDArray}s for equal shapes.
     *
     * <p>Shapes are considered equal if:
     *
     * <ul>
     *   <li>Both {@link MxNDArray}s have equal rank, and
     *   <li>size(0)...size(rank()-1) are equal for both {@link MxNDArray}s
     * </ul>
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.ones(new Shape(1, 2, 3));
     * jshell&gt; MxNDArray array2 = manager.create(new Shape(1, 2, 3));
     * jshell&gt; MxNDArrays.shapeEquals(array1, array2); // return true instead of boolean MxNDArray
     * true
     * </pre>
     *
     * @param a the {@link MxNDArray} to compare
     * @param b the {@link MxNDArray} to compare
     * @return {@code true} if the {@link Shape}s are the same
     */
    public static boolean shapeEquals(MxNDArray a, MxNDArray b) {
        return a.shapeEquals(b);
    }

    /**
     * Returns {@code true} if two {@link MxNDArray} are element-wise equal within a tolerance.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new double[] {1e10,1e-7});
     * jshell&gt; MxNDArray array2 = manager.create(new double[] {1.00001e10,1e-8});
     * jshell&gt; MxNDArrays.allClose(array1, array2); // return false instead of boolean MxNDArray
     * false
     * jshell&gt; MxNDArray array1 = manager.create(new double[] {1e10,1e-8});
     * jshell&gt; MxNDArray array2 = manager.create(new double[] {1.00001e10,1e-9});
     * jshell&gt; MxNDArrays.allClose(array1, array2); // return true instead of boolean MxNDArray
     * true
     * </pre>
     *
     * @param a the {@link MxNDArray} to compare with
     * @param b the {@link MxNDArray} to compare with
     * @return the boolean result
     */
//    public static boolean allClose(MxNDArray a, MxNDArray b) {
//        return a.allClose(b);
//    }

    /**
     * Returns {@code true} if two {@link MxNDArray} are element-wise equal within a tolerance.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new double[] {1e10, 1e-7});
     * jshell&gt; MxNDArray array2 = manager.create(new double[] {1.00001e10, 1e-8});
     * jshell&gt; MxNDArrays.allClose(array1, array2, 1e-05, 1e-08, false); // return false instead of boolean MxNDArray
     * false
     * jshell&gt; MxNDArray array1 = manager.create(new double[] {1e10, 1e-8});
     * jshell&gt; MxNDArray array2 = manager.create(new double[] {1.00001e10, 1e-9});
     * jshell&gt; MxNDArrays.allClose(array1, array2, 1e-05, 1e-08, false); // return true instead of boolean MxNDArray
     * true
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {1f, Float.NaN});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {1f, Float.NaN});
     * jshell&gt; MxNDArrays.allClose(array1, array2, 1e-05, 1e-08, true); // return true instead of boolean MxNDArray
     * true
     * </pre>
     *
     * @param a the {@link MxNDArray} to compare with
     * @param b the {@link MxNDArray} to compare with
     * @param rtol the relative tolerance parameter
     * @param atol the absolute tolerance parameter
     * @param equalNan whether to compare NaN’s as equal. If {@code true}, NaN’s in the {@link
     *     MxNDArray} will be considered equal to NaN’s in the other {@link MxNDArray}
     * @return the boolean result
     */
//    public static boolean allClose(
//            MxNDArray a, MxNDArray b, double rtol, double atol, boolean equalNan) {
//        return a.allClose(b, rtol, atol, equalNan);
//    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.ones(new Shape(1));
     * jshell&gt; MxNDArrays.eq(array, 1);
     * ND: (1) cpu() boolean
     * [ true]
     * </pre>
     *
     * @param a the {@link MxNDArray} to compare
     * @param n the number to compare
     * @return the boolean {@link MxNDArray} for element-wise "Equals" comparison
     */
    public static MxNDArray eq(MxNDArray a, Number n) {
        return a.eq(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.ones(new Shape(1));
     * jshell&gt; MxNDArrays.eq(1, array);
     * ND: (1) cpu() boolean
     * [ true]
     * </pre>
     *
     * @param n the number to compare
     * @param a the {@link MxNDArray} to compare
     * @return the boolean {@link MxNDArray} for element-wise "Equals" comparison
     */
    public static MxNDArray eq(Number n, MxNDArray a) {
        return a.eq(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {0f, 1f, 3f});
     * jshell&gt; MxNDArray array2 = manager.arange(3f);
     * jshell&gt; MxNDArrays.eq(array1, array2);
     * ND: (3) cpu() boolean
     * [ true,  true, false]
     * </pre>
     *
     * @param a the {@link MxNDArray} to compare
     * @param b the {@link MxNDArray} to compare
     * @return the boolean {@link MxNDArray} for element-wise "Equals" comparison
     */
    public static MxNDArray eq(MxNDArray a, MxNDArray b) {
        return a.eq(b);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Not equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(4f).reshape(2, 2);
     * jshell&gt; MxNDArrays.neq(array, 1);
     * ND: (2, 2) cpu() boolean
     * [[ true, false],
     *  [ true,  true],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to compare
     * @param n the number to compare
     * @return the boolean {@link MxNDArray} for element-wise "Not equals" comparison
     */
    public static MxNDArray neq(MxNDArray a, Number n) {
        return a.neq(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Not equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(f4).reshape(2, 2);
     * jshell&gt; MxNDArrays.neq(1, array);
     * ND: (2, 2) cpu() boolean
     * [[ true, false],
     *  [ true,  true],
     * ]
     * </pre>
     *
     * @param n the number to compare
     * @param a the {@link MxNDArray} to compare
     * @return the boolean {@link MxNDArray} for element-wise "Not equals" comparison
     */
    public static MxNDArray neq(Number n, MxNDArray a) {
        return a.neq(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Not equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {1f, 3f});
     * jshell&gt; MxNDArrays.neq(array1, array2);
     * ND: (2) cpu() boolean
     * [false,  true]
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {1f, 3f, 1f, 4f}, new Shape(2, 2));
     * jshell&gt; MxNDArrays.neq(array1, array2); // broadcasting
     * ND: (2, 2) cpu() boolean
     * [[false,  true],
     *  [false,  true],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to compare
     * @param b the {@link MxNDArray} to compare
     * @return the boolean {@link MxNDArray} for element-wise "Not equals" comparison
     */
    public static MxNDArray neq(MxNDArray a, MxNDArray b) {
        return a.neq(b);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Greater Than" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {4f, 2f});
     * jshell&gt; MxNDArrays.gt(array, 2f);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * @param a the {@link MxNDArray} to compare
     * @param n the number to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Greater Than" comparison
     */
    public static MxNDArray gt(MxNDArray a, Number n) {
        return a.gt(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Greater Than" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {4f, 2f});
     * jshell&gt; MxNDArrays.gt(2f, array);
     * ND: (2) cpu() boolean
     * [false, false]
     * </pre>
     *
     * @param n the number to be compared
     * @param a the MxNDArray to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Greater Than" comparison
     */
    public static MxNDArray gt(Number n, MxNDArray a) {
        return a.lt(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Greater Than" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {4f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; MxNDArrays.gt(array1, array2);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param b the {@link MxNDArray} to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Greater Than" comparison
     */
    public static MxNDArray gt(MxNDArray a, MxNDArray b) {
        return a.gt(b);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Greater or equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {4f, 2f});
     * jshell&gt; MxNDArrays.gte(array, 2);
     * ND: (2) cpu() boolean
     * [ true, true]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param n the number to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Greater or equals" comparison
     */
    public static MxNDArray gte(MxNDArray a, Number n) {
        return a.gte(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Greater or equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {4f, 2f});
     * jshell&gt; MxNDArrays.gte(2, array);
     * ND: (2) cpu() boolean
     * [false,  true]
     * </pre>
     *
     * @param n the number to be compared
     * @param a the {@link MxNDArray} to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Greater or equals" comparison
     */
    public static MxNDArray gte(Number n, MxNDArray a) {
        return a.lte(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Greater or equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {4f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; MxNDArrays.gte(array1, array2);
     * ND: (2) cpu() boolean
     * [ true, true]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param b the {@link MxNDArray} to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Greater or equals" comparison
     */
    public static MxNDArray gte(MxNDArray a, MxNDArray b) {
        return a.gte(b);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Less" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.lt(array, 2f);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param n the number to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Less" comparison
     */
    public static MxNDArray lt(MxNDArray a, Number n) {
        return a.lt(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Less" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.lt(2f, array);
     * ND: (2) cpu() boolean
     * [false, false]
     * </pre>
     *
     * @param n the number to be compared
     * @param a the {@link MxNDArray} to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Less" comparison
     */
    public static MxNDArray lt(Number n, MxNDArray a) {
        return a.gt(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Less" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; MxNDArrays.lt(array1, array2);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param b the {@link MxNDArray} to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Less" comparison
     */
    public static MxNDArray lt(MxNDArray a, MxNDArray b) {
        return a.lt(b);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Less or equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.lte(array, 2f);
     * ND: (2) cpu() boolean
     * [ true, true]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param n the number to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Less or equals" comparison
     */
    public static MxNDArray lte(MxNDArray a, Number n) {
        return a.lte(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Less or equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.lte(2f, array);
     * ND: (2) cpu() boolean
     * [false,  true]
     * </pre>
     *
     * @param n the number to be compared
     * @param a the {@link MxNDArray} to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Less or equals" comparison
     */
    public static MxNDArray lte(Number n, MxNDArray a) {
        return a.gte(n);
    }

    /**
     * Returns the boolean {@link MxNDArray} for element-wise "Less or equals" comparison.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; MxNDArrays.lte(array1, array2)
     * ND: (2) cpu() boolean
     * [ true, true]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param b the {@link MxNDArray} to be compared against
     * @return the boolean {@link MxNDArray} for element-wise "Less or equals" comparison
     */
    public static MxNDArray lte(MxNDArray a, MxNDArray b) {
        return a.lte(b);
    }

    /**
     * Returns elements chosen from the {@link MxNDArray} or the other {@link MxNDArray} depending on
     * condition.
     *
     * <p>Given three {@link MxNDArray}s, condition, a, and b, returns an {@link MxNDArray} with the
     * elements from a or b, depending on whether the elements from condition {@link MxNDArray} are
     * {@code true} or {@code false}. If condition has the same shape as a, each element in the
     * output {@link MxNDArray} is from this if the corresponding element in the condition is {@code
     * true}, and from other if {@code false}.
     *
     * <p>Note that all non-zero values are interpreted as {@code true} in condition {@link
     * MxNDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(10f);
     * jshell&gt; MxNDArrays.where(array.lt(5), array, array.mul(10));
     * ND: (10) cpu() float32
     * [ 0.,  1.,  2.,  3.,  4., 50., 60., 70., 80., 90.]
     * jshell&gt; MxNDArray array = manager.create(new float[]{0f, 1f, 2f, 0f, 2f, 4f, 0f, 3f, 6f}, new Shape(3, 3));
     * jshell&gt; MxNDArrays.where(array.lt(4), array, manager.create(-1f));
     * ND: (3, 3) cpu() float32
     * [[ 0.,  1.,  2.],
     *  [ 0.,  2., -1.],
     *  [ 0.,  3., -1.],
     * ]
     * </pre>
     *
     * @param condition the condition {@code MxNDArray}
     * @param a the first {@link MxNDArray}
     * @param b the other {@link MxNDArray}
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray where(MxNDArray condition, MxNDArray a, MxNDArray b) {
        return a.getNDArrayInternal().where(condition, b);
    }

    /**
     * Returns the maximum of a {@link MxNDArray} and a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; MxNDArrays.maximum(array, 3f);
     * ND: (3) cpu() float32
     * [3., 3., 4.]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param n the number to be compared
     * @return the maximum of a {@link MxNDArray} and a number element-wise
     */
    public static MxNDArray maximum(MxNDArray a, Number n) {
        return a.maximum(n);
    }

    /**
     * Returns the maximum of a number and a {@link MxNDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; MxNDArrays.maximum(3f, array);
     * ND: (3) cpu() float32
     * [3., 3., 4.]
     * </pre>
     *
     * @param n the number to be compared
     * @param a the {@link MxNDArray} to be compared
     * @return the maximum of a number and a {@link MxNDArray} element-wise
     */
    public static MxNDArray maximum(Number n, MxNDArray a) {
        return maximum(a, n);
    }

    /**
     * Returns the maximum of {@link MxNDArray} a and {@link MxNDArray} b element-wise.
     *
     * <p>The shapes of {@link MxNDArray} a and {@link MxNDArray} b must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {1f, 5f, 2f});
     * jshell&gt; MxNDArrays.maximum(array1, array2);
     * ND: (3) cpu() float32
     * [2., 5., 4.]
     * jshell&gt; MxNDArray array1 = manager.eye(2);
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {0.5f, 2f});
     * jshell&gt; MxNDArrays.maximum(array1, array2); // broadcasting
     * ND: (2, 2) cpu() float32
     * [[1. , 2. ],
     *  [0.5, 2. ],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param b the {@link MxNDArray} to be compared
     * @return the maximum of {@link MxNDArray} a and {@link MxNDArray} b element-wise
     */
    public static MxNDArray maximum(MxNDArray a, MxNDArray b) {
        return a.maximum(b);
    }

    /**
     * Returns the minimum of a {@link MxNDArray} and a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; MxNDArrays.minimum(array, 3f);
     * ND: (3) cpu() float32
     * [2., 3., 3.]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param n the number to be compared
     * @return the minimum of a {@link MxNDArray} and a number element-wise
     */
    public static MxNDArray minimum(MxNDArray a, Number n) {
        return a.minimum(n);
    }

    /**
     * Returns the minimum of a number and a {@link MxNDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; MxNDArrays.minimum(3f, array);
     * ND: (3) cpu() float32
     * [2., 3., 3.]
     * </pre>
     *
     * @param n the number to be compared
     * @param a the {@link MxNDArray} to be compared
     * @return the minimum of a number and a {@link MxNDArray} element-wise
     */
    public static MxNDArray minimum(Number n, MxNDArray a) {
        return minimum(a, n);
    }

    /**
     * Returns the minimum of {@link MxNDArray} a and {@link MxNDArray} b element-wise.
     *
     * <p>The shapes of {@link MxNDArray} a and {@link MxNDArray} b must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {1f, 5f, 2f});
     * jshell&gt; MxNDArrays.minimum(array1, array2);
     * ND: (3) cpu() float32
     * [1., 3., 2.]
     * jshell&gt; MxNDArray array1 = manager.eye(2);
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {0.5f, 2f});
     * jshell&gt; MxNDArrays.minimum(array1, array2); // broadcasting
     * ND: (2, 2) cpu() float32
     * [[0.5, 0. ],
     *  [0. , 1. ],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be compared
     * @param b the {@link MxNDArray} to be compared
     * @return the minimum of {@link MxNDArray} a and {@link MxNDArray} b element-wise
     */
    public static MxNDArray minimum(MxNDArray a, MxNDArray b) {
        return a.minimum(b);
    }

    /**
     * Returns portion of the {@link MxNDArray} given the index boolean {@link MxNDArray} along first
     * axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(3, 2));
     * jshell&gt; MxNDArray mask = manager.create(new boolean[] {true, false, true});
     * jshell&gt; MxNDArrays.booleanMask(array, mask);
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     *  [5., 6.],
     * ]
     * </pre>
     *
     * @param data the {@link MxNDArray} to operate on
     * @param index the boolean {@link MxNDArray} mask
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray booleanMask(MxNDArray data, MxNDArray index) {
        return booleanMask(data, index, 0);
    }

    /**
     * Returns portion of the {@link MxNDArray} given the index boolean {@link MxNDArray} along given
     * axis.
     *
     * @param data the {@link MxNDArray} to operate on
     * @param index the boolean {@link MxNDArray} mask
     * @param axis an integer that represents the axis of {@link MxNDArray} to mask from
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray booleanMask(MxNDArray data, MxNDArray index, int axis) {
        return data.booleanMask(index, axis);
    }

    /**
     * Sets all elements of the given {@link MxNDArray} outside the sequence {@link MxNDArray} to a
     * constant value.
     *
     * <p>This function takes an n-dimensional input array of the form [batch_size,
     * max_sequence_length, ....] and returns an array of the same shape. Parameter {@code
     * sequenceLength} is used to handle variable-length sequences. {@code sequenceLength} should be
     * an input array of positive ints of dimension [batch_size].
     *
     * @param data the {@link MxNDArray} to operate on
     * @param sequenceLength used to handle variable-length sequences
     * @param value the constant value to be set
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray sequenceMask(MxNDArray data, MxNDArray sequenceLength, float value) {
        return data.sequenceMask(sequenceLength, value);
    }

    /**
     * Sets all elements of the given {@link MxNDArray} outside the sequence {@link MxNDArray} to 0.
     *
     * <p>This function takes an n-dimensional input array of the form [batch_size,
     * max_sequence_length, ....] and returns an array of the same shape. Parameter {@code
     * sequenceLength} is used to handle variable-length sequences. {@code sequenceLength} should be
     * an input array of positive ints of dimension [batch_size].
     *
     * @param data the {@link MxNDArray} to operate on
     * @param sequenceLength used to handle variable-length sequences
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray sequenceMask(MxNDArray data, MxNDArray sequenceLength) {
        return data.sequenceMask(sequenceLength);
    }

    ////////////////////////////////////////
    // Operations: Element Arithmetic
    ////////////////////////////////////////

    /**
     * Adds a number to the {@link MxNDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.add(array, 2f);
     * ND: (2) cpu() float32
     * [3., 4.]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be added to
     * @param n the number to add
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray add(MxNDArray a, Number n) {
        return a.add(n);
    }

    /**
     * Adds a {@link MxNDArray} to a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.add(2f, array);
     * ND: (2) cpu() float32
     * [3., 4.]
     * </pre>
     *
     * @param n the number to be added to
     * @param a the {@link MxNDArray} to add
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray add(Number n, MxNDArray a) {
        return a.add(n);
    }

    /**
     * Adds a {@link MxNDArray} to a {@link MxNDArray} element-wise.
     *
     * <p>The shapes of all of the {@link MxNDArray}s must be the same.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.add(array, array, array);
     * ND: (2) cpu() float32
     * [3., 6.]
     * </pre>
     *
     * @param arrays the {@link MxNDArray}s to add together
     * @return the result {@link MxNDArray}
     * @throws IllegalArgumentException arrays must have at least two elements
     * @throws IllegalArgumentException the shape of all inputs must be the same
     */
    public static MxNDArray add(MxNDArray... arrays) {
        checkInputs(arrays);
        if (arrays.length == 2) {
            return arrays[0].add(arrays[1]);
        }
        try (MxNDArray array = MxNDArrays.stack(new MxNDList(arrays))) {
            return array.sum(new int[] {0});
        }
    }

    /**
     * Subtracts a number from the {@link MxNDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.sub(2f);
     * ND: (2) cpu() float32
     * [-1.,  0.]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be subtracted
     * @param n the number to subtract from
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray sub(MxNDArray a, Number n) {
        return a.sub(n);
    }

    /**
     * Subtracts a {@link MxNDArray} from a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.sub(3f, array);
     * ND: (2) cpu() float32
     * [2., 1.]
     * </pre>
     *
     * @param n the number to be subtracted
     * @param a the {@link MxNDArray} to subtract from
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray sub(Number n, MxNDArray a) {
        return a.getNDArrayInternal().rsub(n);
    }

    /**
     * Subtracts a {@link MxNDArray} from a {@link MxNDArray} element-wise.
     *
     * <p>The shapes of {@link MxNDArray} a and {@link MxNDArray} b must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; MxNDArray array2 = manager.arange(3f);
     * jshell&gt; MxNDArrays.sub(array1, array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0., 0., 0.],
     *  [3., 3., 3.],
     *  [6., 6., 6.],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be subtracted
     * @param b the {@link MxNDArray} to subtract from
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray sub(MxNDArray a, MxNDArray b) {
        return a.sub(b);
    }

    /**
     * Multiplies the {@link MxNDArray} by a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.mul(array, 3f);
     * ND: (2) cpu() float32
     * [3., 6.]
     * </pre>
     *
     * @param a the MxNDArray to be multiplied
     * @param n the number to multiply by
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray mul(MxNDArray a, Number n) {
        return a.mul(n);
    }

    /**
     * Multiplies a number by a {@link MxNDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.mul(3f, array);
     * ND: (2) cpu() float32
     * [3., 6.]
     * </pre>
     *
     * @param n the number to be multiplied
     * @param a the {@link MxNDArray} to multiply by
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray mul(Number n, MxNDArray a) {
        return a.mul(n);
    }

    /**
     * Multiplies all of the {@link MxNDArray}s together element-wise.
     *
     * <p>The shapes of all of the {@link MxNDArray}s must be the same.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.mul(array, array, array);
     * ND: (2) cpu() float32
     * [1., 8.]
     * </pre>
     *
     * @param arrays the {@link MxNDArray}s to multiply together
     * @return the result {@link MxNDArray}
     * @throws IllegalArgumentException arrays must have at least two elements
     * @throws IllegalArgumentException the shape of all inputs must be the same
     */
    public static MxNDArray mul(MxNDArray... arrays) {
        checkInputs(arrays);
        if (arrays.length == 2) {
            return arrays[0].mul(arrays[1]);
        }
        try (MxNDArray array = MxNDArrays.stack(new MxNDList(arrays))) {
            return array.prod(new int[] {0});
        }
    }

    /**
     * Divides the {@link MxNDArray} by a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(5f);
     * jshell&gt; MxNDArrays.div(array, 4f);
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be be divided
     * @param n the number to divide by
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray div(MxNDArray a, Number n) {
        return a.div(n);
    }

    /**
     * Divides a number by a {@link MxNDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(5f).add(1);
     * jshell&gt; MxNDArrays.div(4f, array);
     * ND: (5) cpu() float32
     * [4.    , 2.    , 1.3333, 1.    , 0.8   ]
     * </pre>
     *
     * @param n the number to be be divided
     * @param a the {@link MxNDArray} to divide by
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray div(Number n, MxNDArray a) {
        return a.getNDArrayInternal().rdiv(n);
    }

    /**
     * Divides a {@link MxNDArray} by a {@link MxNDArray} element-wise.
     *
     * <p>The shapes of {@link MxNDArray} a and {@link MxNDArray} b must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; MxNDArray array2 = manager.ones(new Shape(3)).mul(10);
     * jshell&gt; MxNDArrays.div(array1, array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0. , 0.1, 0.2],
     *  [0.3, 0.4, 0.5],
     *  [0.6, 0.7, 0.8],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be be divided
     * @param b the {@link MxNDArray} to divide by
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray div(MxNDArray a, MxNDArray b) {
        return a.div(b);
    }

    /**
     * Returns element-wise remainder of division.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(7f);
     * jshell&gt; MxNDArrays.mod(array, 5f);
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
     * </pre>
     *
     * @param a the dividend {@link MxNDArray}
     * @param n the divisor number
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray mod(MxNDArray a, Number n) {
        return a.mod(n);
    }

    /**
     * Returns element-wise remainder of division.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(7f).add(1);
     * jshell&gt; MxNDArrays.mod(5f, array);
     * ND: (7) cpu() float32
     * [0., 1., 2., 1., 0., 5., 5.]
     * </pre>
     *
     * @param n the dividend number
     * @param a the divisor {@link MxNDArray}
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray mod(Number n, MxNDArray a) {
        return a.getNDArrayInternal().rmod(n);
    }

    /**
     * Returns element-wise remainder of division.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {4f, 7f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; MxNDArrays.mod(array1, array2);
     * ND: (2) cpu() float32
     * [0., 1.]
     * </pre>
     *
     * @param a the dividend MxNDArray
     * @param b the dividend MxNDArray
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray mod(MxNDArray a, MxNDArray b) {
        return a.mod(b);
    }

    /**
     * Takes the power of the {@link MxNDArray} with a number element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(5f);
     * jshell&gt; MxNDArrays.pow(array, 4f);
     * ND: (6) cpu() float32
     * [  0.,   1.,   8.,  27.,  64., 125.]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be taken the power with
     * @param n the number to take the power with
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray pow(MxNDArray a, Number n) {
        return a.pow(n);
    }

    /**
     * Takes the power of a number with a {@link MxNDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(5f);
     * jshell&gt; MxNDArrays.pow(4f, array);
     * ND: (5) cpu() float32
     * [  1.,   4.,  16.,  64., 256.]
     * </pre>
     *
     * @param n the number to be taken the power with
     * @param a the {@link MxNDArray} to take the power with
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray pow(Number n, MxNDArray a) {
        return a.getNDArrayInternal().rpow(n);
    }

    /**
     * Takes the power of a {@link MxNDArray} with a {@link MxNDArray} element-wise.
     *
     * <p>The shapes of {@link MxNDArray} a and {@link MxNDArray} b must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.arange(6f).reshape(3, 2);
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; MxNDArrays.pow(array1, array2); // broadcasting
     * ND: (3, 2) cpu() float32
     * [[  0.,   1.],
     *  [  4.,  27.],
     *  [ 16., 125.],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be taken the power with
     * @param b the {@link MxNDArray} to take the power with
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray pow(MxNDArray a, MxNDArray b) {
        return a.pow(b);
    }

    /**
     * Adds a number to the {@link MxNDArray} element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.addi(array, 2f);
     * ND: (2) cpu() float32
     * [3., 4.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [3., 4.]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be added to
     * @param n the number to add
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray addi(MxNDArray a, Number n) {
        return a.addi(n);
    }

    /**
     * Adds a {@link MxNDArray} to a number element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.addi(2f, array);
     * ND: (2) cpu() float32
     * [3., 4.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [3., 4.]
     * </pre>
     *
     * @param a the number to be added to
     * @param n the {@link MxNDArray} to add
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray addi(Number n, MxNDArray a) {
        return a.addi(n);
    }

    /**
     * Adds all of the {@link MxNDArray}s together element-wise in place.
     *
     * <p>The shapes of all of the {@link MxNDArray}s must be the same.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {3f, 4f});
     * jshell&gt; MxNDArray array3 = manager.create(new float[] {5f, 6f});
     * jshell&gt; MxNDArrays.addi(array1, array2, array3);
     * ND: (2) cpu() float32
     * [9., 12.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [9., 12.]
     * </pre>
     *
     * @param arrays the {@link MxNDArray}s to add together
     * @return the result {@link MxNDArray}
     * @throws IllegalArgumentException arrays must have at least two elements
     */
    public static MxNDArray addi(MxNDArray... arrays) {
        checkInputs(arrays);
        Arrays.stream(arrays).skip(1).forEachOrdered(array -> arrays[0].addi(array));
        return arrays[0];
    }

    /**
     * Subtracts a number from the {@link MxNDArray} element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.subi(array, 2f);
     * ND: (2) cpu() float32
     * [-1.,  0.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [-1.,  0.]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be subtracted
     * @param n the number to subtract from
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray subi(MxNDArray a, Number n) {
        return a.subi(n);
    }

    /**
     * Subtracts a {@link MxNDArray} from a number element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.subi(3f, array);
     * ND: (2) cpu() float32
     * [2., 1.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [2., 1.]
     * </pre>
     *
     * @param n the number to be subtracted
     * @param a the {@link MxNDArray} to subtract from
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray subi(Number n, MxNDArray a) {
        return a.getNDArrayInternal().rsubi(n);
    }

    /**
     * Subtracts a {@link MxNDArray} from a {@link MxNDArray} element-wise in place.
     *
     * <p>The shapes of {@link MxNDArray} a and {@link MxNDArray} b must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; MxNDArray array2 = manager.arange(3f);
     * jshell&gt; MxNDArrays.subi(array1, array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0., 0., 0.],
     *  [3., 3., 3.],
     *  [6., 6., 6.],
     * ]
     * jshell&gt; array1;
     * [[0., 0., 0.],
     *  [3., 3., 3.],
     *  [6., 6., 6.],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be subtracted
     * @param b the {@link MxNDArray} to subtract from
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray subi(MxNDArray a, MxNDArray b) {
        return a.subi(b);
    }

    /**
     * Multiplies the {@link MxNDArray} by a number element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.muli(array, 3f);
     * ND: (2) cpu() float32
     * [3., 6.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [3., 6.]
     * </pre>
     *
     * @param a the MxNDArray to be multiplied
     * @param n the number to multiply by
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray muli(MxNDArray a, Number n) {
        return a.muli(n);
    }

    /**
     * Multiplies a number by a {@link MxNDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.muli(3f, array);
     * ND: (2) cpu() float32
     * [3., 6.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [3., 6.]
     * </pre>
     *
     * @param n the number to multiply by
     * @param a the {@link MxNDArray} to multiply by
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray muli(Number n, MxNDArray a) {
        return a.muli(n);
    }

    /**
     * Multiplies all of the {@link MxNDArray}s together element-wise in place.
     *
     * <p>The shapes of all of the {@link MxNDArray}s must be the same.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {3f, 4f});
     * jshell&gt; MxNDArray array3 = manager.create(new float[] {5f, 6f});
     * jshell&gt; MxNDArrays.muli(array1, array2, array3);
     * ND: (2) cpu() float32
     * [15., 48.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [15., 48.]
     * </pre>
     *
     * @param arrays the {@link MxNDArray}s to multiply together
     * @return the result {@link MxNDArray}
     * @throws IllegalArgumentException arrays must have at least two elements
     */
    public static MxNDArray muli(MxNDArray... arrays) {
        checkInputs(arrays);
        Arrays.stream(arrays).skip(1).forEachOrdered(array -> arrays[0].muli(array));
        return arrays[0];
    }

    /**
     * Divides a number by a {@link MxNDArray} element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(5f);
     * jshell&gt; MxNDArrays.divi(array, 4f);
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
     * jshell&gt; array;
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be be divided
     * @param n the number to divide by
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray divi(MxNDArray a, Number n) {
        return a.divi(n);
    }

    /**
     * Divides a number by a {@link MxNDArray} element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(5f).add(1);
     * jshell&gt; MxNDArrays.divi(4f, array);
     * ND: (5) cpu() float32
     * [4.    , 2.    , 1.3333, 1.    , 0.8   ]
     * jshell&gt; array;
     * ND: (5) cpu() float32
     * [4.    , 2.    , 1.3333, 1.    , 0.8   ]
     * </pre>
     *
     * @param n the number to be be divided
     * @param a the {@link MxNDArray} to divide by
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray divi(Number n, MxNDArray a) {
        return a.getNDArrayInternal().rdivi(n);
    }

    /**
     * Divides a {@link MxNDArray} by a {@link MxNDArray} element-wise.
     *
     * <p>The shapes of {@link MxNDArray} a and {@link MxNDArray} b must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; MxNDArray array2 = manager.ones(new Shape(3)).mul(10);
     * jshell&gt; MxNDArrays.divi(array1, array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0. , 0.1, 0.2],
     *  [0.3, 0.4, 0.5],
     *  [0.6, 0.7, 0.8],
     * ]
     * jshell&gt; array1;
     * [[0. , 0.1, 0.2],
     *  [0.3, 0.4, 0.5],
     *  [0.6, 0.7, 0.8],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be be divided
     * @param b the {@link MxNDArray} to divide by
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray divi(MxNDArray a, MxNDArray b) {
        return a.divi(b);
    }

    /**
     * Returns element-wise remainder of division in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(7f);
     * jshell&gt; MxNDArrays.modi(array, 5f);
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
     * jshell&gt; array;
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
     * </pre>
     *
     * @param a the dividend {@link MxNDArray}
     * @param n the divisor number
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray modi(MxNDArray a, Number n) {
        return a.modi(n);
    }

    /**
     * Returns element-wise remainder of division in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(7f);
     * jshell&gt; MxNDArrays.modi(5f, array);
     * ND: (7) cpu() float32
     * [0., 0., 1., 2., 1., 0., 5.]
     * jshell&gt; array;
     * ND: (7) cpu() float32
     * [0., 0., 1., 2., 1., 0., 5.]
     * </pre>
     *
     * @param n the dividend number
     * @param a the divisor {@link MxNDArray}
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray modi(Number n, MxNDArray a) {
        return a.getNDArrayInternal().rmodi(n);
    }

    /**
     * Returns element-wise remainder of division.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {4f, 7f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; MxNDArrays.modi(array1, array2);
     * ND: (2) cpu() float32
     * [0., 1.]
     * jshell&gt; array1;
     * ND: (2) cpu() float32
     * [0., 1.]
     * </pre>
     *
     * @param a the dividend MxNDArray
     * @param b the dividend MxNDArray
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray modi(MxNDArray a, MxNDArray b) {
        return a.modi(b);
    }

    /**
     * Takes the power of the {@link MxNDArray} with a number element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(5f);
     * jshell&gt; MxNDArrays.powi(array, 4f);
     * ND: (6) cpu() float32
     * [  0.,   1.,   8.,  27.,  64., 125.]
     * jshell&gt; array;
     * ND: (6) cpu() float32
     * [  0.,   1.,   8.,  27.,  64., 125.]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be taken the power with
     * @param n the number to take the power with
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray powi(MxNDArray a, Number n) {
        return a.powi(n);
    }

    /**
     * Takes the power of a number with a {@link MxNDArray} element-wise in place.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(5f);
     * jshell&gt; MxNDArrays.powi(4f, array);
     * ND: (5) cpu() float32
     * [  1.,   4.,  16.,  64., 256.]
     * jshell&gt; array;
     * ND: (5) cpu() float32
     * [  1.,   4.,  16.,  64., 256.]
     * </pre>
     *
     * @param n the number to be taken the power with
     * @param a the {@link MxNDArray} to take the power with
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray powi(Number n, MxNDArray a) {
        return a.getNDArrayInternal().rpowi(n);
    }

    /**
     * Takes the power of a {@link MxNDArray} with a {@link MxNDArray} element-wise.
     *
     * <p>The shapes of {@link MxNDArray} a and {@link MxNDArray} b must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.arange(6f).reshape(3, 2);
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; MxNDArrays.powi(array1, array2); // broadcasting
     * ND: (3, 2) cpu() float32
     * [[  0.,   1.],
     *  [  4.,  27.],
     *  [ 16., 125.],
     * ]
     * jshell&gt; array1;
     * ND: (3, 2) cpu() float32
     * [[  0.,   1.],
     *  [  4.,  27.],
     *  [ 16., 125.],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to be taken the power with
     * @param b the {@link MxNDArray} to take the power with
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray powi(MxNDArray a, MxNDArray b) {
        return a.powi(b);
    }

    /**
     * Dot product of {@link MxNDArray} a and {@link MxNDArray} b.
     *
     * <ul>
     *   <li>If both the {@link MxNDArray} and the other {@link MxNDArray} are 1-D {@link MxNDArray}s, it
     *       is inner product of vectors (without complex conjugation).
     *   <li>If both the {@link MxNDArray} and the other {@link MxNDArray} are 2-D {@link MxNDArray}s, it
     *       is matrix multiplication.
     *   <li>If either the {@link MxNDArray} or the other {@link MxNDArray} is 0-D {@link MxNDArray}
     *       (scalar), it is equivalent to mul.
     *   <li>If the {@link MxNDArray} is N-D {@link MxNDArray} and the other {@link MxNDArray} is 1-D
     *       {@link MxNDArray}, it is a sum product over the last axis of those.
     *   <li>If the {@link MxNDArray} is N-D {@link MxNDArray} and the other {@link MxNDArray} is M-D
     *       {@link MxNDArray}(where M&gt;&#61;2), it is a sum product over the last axis of this
     *       {@link MxNDArray} and the second-to-last axis of the other {@link MxNDArray}
     * </ul>
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {1f, 2f, 3f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {4f, 5f, 6f});
     * jshell&gt; MxNDArrays.dot(array1, array2); // inner product
     * ND: () cpu() float32
     * 32.
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {5f, 6f, 7f, 8f}, new Shape(2, 2));
     * jshell&gt; MxNDArrays.dot(array1, array2); // matrix multiplication
     * ND: (2, 2) cpu() float32
     * [[19., 22.],
     *  [43., 50.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(5f);
     * jshell&gt; MxNDArrays.dot(array1, array2);
     * ND: (2, 2) cpu() float32
     * [[ 5., 10.],
     *  [15., 20.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.dot(array1, array2);
     * ND: (2) cpu() float32
     * [ 5., 11.]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, new Shape(2, 2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f, 3f ,4f}, new Shape(2, 2));
     * jshell&gt; MxNDArrays.dot(array1, array2);
     * ND: (2, 2, 2) cpu() float32
     * [[[ 7., 10.],
     *   [15., 22.],
     *  ],
     *  [[23., 34.],
     *   [31., 46.],
     *  ],
     * ]
     * </pre>
     *
     * @param a the {@link MxNDArray} to perform dot product with
     * @param b the {@link MxNDArray} to perform dot product with
     * @return the result {@link MxNDArray}
     */
    public static MxNDArray dot(MxNDArray a, MxNDArray b) {
        return a.dot(b);
    }

    /**
     * Product matrix of this {@code MxNDArray} and the other {@code MxNDArray}.
     *
     * <p>The behavior depends on the arguments in the following way.
     *
     * <ul>
     *   <li>If both this {@code MxNDArray} and the other {@code MxNDArray} are 2-D {@code MxNDArray}s,
     *       they are multiplied like conventional matrices
     *   <li>If either this {@code MxNDArray} or the other {@code MxNDArray} is N-D {@code MxNDArray}, N
     *       &gt; 2 , it is treated as a stack of matrices residing in the last two indexes and
     *       broadcast accordingly.
     *   <li>If this {@code MxNDArray} is 1-D {@code MxNDArray}, it is promoted to a matrix by
     *       prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is
     *       removed.
     *   <li>If other {@code MxNDArray} is 1-D {@code MxNDArray}, it is promoted to a matrix by
     *       appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
     * </ul>
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {4f, 1f, 2f, 2f}, new Shape(2, 2));
     * jshell&gt; MxNDArrays.matMul(array1, array2); // for 2-D arrays, it is the matrix product
     * ND: (2, 2) cpu() float32
     * [[4., 1.],
     *  [2., 2.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.matMul(array1, array2);
     * ND: (2) cpu() float32
     * [1., 2.]
     * jshell&gt; array1 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f});
     * jshell&gt; MxNDArrays.matMul(array1, array2);
     * ND: (2) cpu() float32
     * [1., 2.]
     * jshell&gt; array1 = manager.arange(2f * 2f * 4f).reshape(2, 2, 4);
     * jshell&gt; array2 = manager.arange(2f * 2f * 4f).reshape(2, 4, 2);
     * jshell&gt; MxNDArrays.matMul(array1, array2);
     * ND: () cpu() float32
     * 98.
     * </pre>
     *
     * @param a the {@link MxNDArray} to perform matrix product with
     * @param b the {@link MxNDArray} to perform matrix product with
     * @return the result {@code MxNDArray}
     */
    public static MxNDArray matMul(MxNDArray a, MxNDArray b) {
        return a.matMul(b);
    }

    /**
     * Joins a sequence of {@link MxNDArray}s in {@link MxNDList} along the first axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {3f, 4f, 5f});
     * jshell&gt; MxNDArray array3 = manager.create(new float[] {6f, 7f, 8f});
     * jshell&gt; MxNDArrays.stack(new MxNDList(array1, array2, array3));
     * ND: (3, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     *  [6., 7., 8.],
     * ]
     * </pre>
     *
     * @param arrays the input {@link MxNDList}. Each {@link MxNDArray} in the {@link MxNDList} must have
     *     the same shape as the {@link MxNDArray}
     * @return the result {@link MxNDArray}. The stacked {@link MxNDArray} has one more dimension than
     *     the {@link MxNDArray}s in {@link MxNDList}
     */
    public static MxNDArray stack(MxNDList arrays) {
        return stack(arrays, 0);
    }

    /**
     * Joins a sequence of {@link MxNDArray}s in {@link MxNDList} along a new axis.
     *
     * <p>The axis parameter specifies the index of the new axis in the dimensions of the result.
     * For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last
     * dimension.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {3f, 4f, 5f});
     * jshell&gt; MxNDArrays.stack(new MxNDList(array1, array2), 0);
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     *  [3., 4., 5.],
     * ]
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {3f, 4f, 5f});
     * jshell&gt; MxNDArrays.stack(new MxNDList(array1, array2), 1);
     * ND: (3, 2) cpu() float32
     * [[0., 3.],
     *  [1., 4.],
     *  [2., 5.],
     * ]
     * </pre>
     *
     * @param arrays the input {@link MxNDList}. Each {@link MxNDArray} in the {@link MxNDList} must have
     *     the same shape as the {@link MxNDArray}
     * @param axis the axis in the result {@link MxNDArray} along which the input {@link MxNDList} are
     *     stacked
     * @return the result {@link MxNDArray}. The stacked {@link MxNDArray} has one more dimension than
     *     the the {@link MxNDArray}
     */
    public static MxNDArray stack(MxNDList arrays, int axis) {
        if (arrays.size() <= 0) {
            throw new IllegalArgumentException("need at least one array to stack");
        }
        MxNDArray array = arrays.head();
        return array.getNDArrayInternal().stack(arrays.subNDList(1), axis);
    }

    /**
     * Joins a {@link MxNDList} along the first axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {3f, 4f, 5f});
     * jshell&gt; MxNDArray array3 = manager.create(new float[] {6f, 7f, 8f});
     * jshell&gt; MxNDArrays.concat(new MxNDList(array1, array2, array3));
     * ND: (9) cpu() float32
     * [0., 1., 2., 3., 4., 5., 6., 7., 8.]
     * </pre>
     *
     * @param arrays a {@link MxNDList} which have the same shape as the {@link MxNDArray}, except in
     *     the dimension corresponding to axis
     * @return the concatenated {@link MxNDArray}
     */
    public static MxNDArray concat(MxNDList arrays) {
        return concat(arrays, 0);
    }

    /**
     * Joins a {@link MxNDList} along an existing axis.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; MxNDArray array2 = manager.create(new float[] {5f, 6f}, new Shape(1, 2));
     * jshell&gt; MxNDArrays.concat(new MxNDList(array1, array2), 0);
     * ND: (3, 2) cpu() float32
     * [[1., 2.],
     *  [3., 4.],
     *  [5., 6.],
     * ]
     * jshell&gt; MxNDArrays.concat(new MxNDList(array1, array2.transpose()), 1);
     * ND: (2, 3) cpu() float32
     * [[1., 2., 5.],
     *  [3., 4., 6.],
     * ]
     * </pre>
     *
     * @param arrays a {@link MxNDList} which have the same shape as the {@link MxNDArray}, except in
     *     the dimension corresponding to axis
     * @param axis the axis along which the {@link MxNDList} will be joined
     * @return the concatenated {@link MxNDArray}
     */
    public static MxNDArray concat(MxNDList arrays, int axis) {

        if (arrays.size() <= 0) {
            throw new IllegalArgumentException("need at least one array to concatenate");
        }

        if (arrays.size() == 1) {
            return arrays.singletonOrThrow().duplicate();
        }
        MxNDArray array = arrays.head();
        return array.getNDArrayInternal().concat(arrays.subNDList(1), axis);
    }

    /**
     * Returns the truth value of {@link MxNDArray} a AND {@link MxNDArray} b element-wise.
     *
     * <p>The shapes of {@link MxNDArray} a and {@link MxNDArray} b must be broadcastable.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new boolean[] {true});
     * jshell&gt; MxNDArray array2 = manager.create(new boolean[] {false});
     * jshell&gt; MxNDArrays.logicalAnd(array1, array2);
     * ND: (1) cpu() boolean
     * [false]
     * jshell&gt; array1 = manager.create(new boolean[] {true, false});
     * jshell&gt; array2 = manager.create(new boolean[] {false, false});
     * jshell&gt; MxNDArrays.logicalAnd(array.gt(1), array.lt(4));
     * ND: (2) cpu() boolean
     * [false, false]
     * </pre>
     *
     * @param a the {@link MxNDArray} to operate on
     * @param b the {@link MxNDArray} to operate on
     * @return the boolean {@link MxNDArray} of the logical AND operation applied to the elements of
     *     the {@link MxNDArray} a and {@link MxNDArray} b
     */
    public static MxNDArray logicalAnd(MxNDArray a, MxNDArray b) {
        return a.logicalAnd(b);
    }

    /**
     * Computes the truth value of {@link MxNDArray} a AND {@link MxNDArray} b element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array1 = manager.create(new boolean[] {true});
     * jshell&gt; MxNDArray array2 = manager.create(new boolean[] {false});
     * jshell&gt; MxNDArrays.logicalOr(array1, array2);
     * ND: (1) cpu() boolean
     * [ true]
     * jshell&gt; array1 = manager.create(new boolean[] {true, false});
     * jshell&gt; array2 = manager.create(new boolean[] {false, false});
     * jshell&gt; MxNDArrays.logicalOr(array1, array2);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(5f);
     * jshell&gt; MxNDArrays.logicalOr(array.lt(1), array.gt(3));
     * ND: (5) cpu() boolean
     * [ true, false, false, false,  true]
     * </pre>
     *
     * @param a the {@link MxNDArray} to operate on
     * @param b the {@link MxNDArray} to operate on
     * @return the boolean {@link MxNDArray} of the logical AND operation applied to the elements of
     *     the {@link MxNDArray} a and {@link MxNDArray} b
     */
    public static MxNDArray logicalOr(MxNDArray a, MxNDArray b) {
        return a.logicalOr(b);
    }

    /**
     * Computes the truth value of {@link MxNDArray} a AND {@link MxNDArray} b element-wise.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new boolean[] {true});
     * jshell&gt; MxNDArrays.logicalXor(array1, array2);
     * ND: (1) cpu() boolean
     * [ true]
     * jshell&gt; array1 = manager.create(new boolean[] {true, false});
     * jshell&gt; array2 = manager.create(new boolean[] {false, false});
     * jshell&gt; MxNDArrays.logicalXor(array1, array2);
     * ND: (2) cpu() boolean
     * [ true, false]
     * </pre>
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.arange(5f);
     * jshell&gt; MxNDArrays.logicalXor(array.lt(1), array.gt(3));
     * ND: (5) cpu() boolean
     * [ true, false, false, false,  true]
     * </pre>
     *
     * @param a the {@link MxNDArray} to operate on
     * @param b the {@link MxNDArray} to operate on
     * @return the boolean {@link MxNDArray} of the logical XOR operation applied to the elements of
     *     the {@link MxNDArray} a and {@link MxNDArray} b
     */
    public static MxNDArray logicalXor(MxNDArray a, MxNDArray b) {
        return a.logicalXor(b);
    }

    /**
     * Returns element-wise inverse gauss error function of the input {@code MxNDArray}.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; MxNDArray array = manager.create(new float[] {0f, 0.5f, -1f});
     * jshell&gt; MxNDArrays.erfinv(array);
     * ND: (3) cpu() float32
     * [0., 0.4769, -inf]
     * </pre>
     *
     * @param input The input {@code MxNDArray}
     * @return The inverse of gauss error of the input, element-wise
     */
    public static MxNDArray erfinv(MxNDArray input) {
        return input.erfinv();
    }
}