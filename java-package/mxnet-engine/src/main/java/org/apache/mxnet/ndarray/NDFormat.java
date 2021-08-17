/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet.ndarray;

import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.util.Utils;

/** A helper for printing an {@link NDArray}. */
public abstract class NDFormat {

    private static final int PRECISION = 8;
    private static final String LF = System.getProperty("line.separator");
    private static final Pattern PATTERN = Pattern.compile("\\s*\\d\\.(\\d*?)0*e[+-](\\d+)");

    /**
     * Formats the contents of an array as a pretty printable string.
     *
     * @param array the array to print
     * @param maxSize the maximum elements to print out
     * @param maxDepth the maximum depth to print out
     * @param maxRows the maximum rows to print out
     * @param maxColumns the maximum columns to print out
     * @return the string representation of the array
     */
    public static String format(
            NDArray array, int maxSize, int maxDepth, int maxRows, int maxColumns) {
        NDFormat format;
        DataType dataType = array.getDataType();

        if (dataType == DataType.UINT8) {
            format = new HexFormat();
        } else if (dataType == DataType.BOOLEAN) {
            format = new BooleanFormat();
        } else if (dataType.isInteger()) {
            format = new IntFormat(array);
        } else {
            format = new FloatFormat(array);
        }
        return format.dump(array, maxSize, maxDepth, maxRows, maxColumns);
    }

    protected abstract CharSequence format(Number value);

    private String dump(NDArray array, int maxSize, int maxDepth, int maxRows, int maxColumns) {
        StringBuilder sb = new StringBuilder(1000);
        String name = array.getName();
        if (name != null) {
            sb.append(name).append(": ");
        } else {
            sb.append("ND: ");
        }
        sb.append(array.getShape())
                .append(' ')
                .append(array.getDevice())
                .append(' ')
                .append(array.getDataType());
        if (array.hasGradient()) {
            sb.append(" hasGradient");
        }
        sb.append(LF);

        long size = array.size();
        long dimension = array.getShape().dimension();
        if (size == 0) {
            // corner case: 0 dimension
            sb.append("[]").append(LF);
        } else if (dimension == 0) {
            // scalar case
            sb.append(format(array.toArray()[0])).append(LF);
        } else if (size > maxSize) {
            sb.append("[ Exceed max print size ]");
        } else if (dimension > maxDepth) {
            sb.append("[ Exceed max print dimension ]");
        } else {
            dump(sb, array, 0, true, maxRows, maxColumns);
        }
        return sb.toString();
    }

    private void dump(
            StringBuilder sb,
            NDArray array,
            int depth,
            boolean first,
            int maxRows,
            int maxColumns) {
        if (!first) {
            Utils.pad(sb, ' ', depth);
        }
        sb.append('[');
        Shape shape = array.getShape();
        if (shape.dimension() == 1) {
            append(sb, array.toArray(), maxColumns);
        } else {
            long len = shape.head();
            long limit = Math.min(len, maxRows);
            for (int i = 0; i < limit; ++i) {
                try (NDArray nd = array.get(i)) {
                    dump(sb, nd, depth + 1, i == 0, maxRows, maxColumns);
                }
            }
            long remaining = len - limit;
            if (remaining > 0) {
                Utils.pad(sb, ' ', depth + 1);
                sb.append("... ").append(remaining).append(" more");
            }
            Utils.pad(sb, ' ', depth);
        }
        // last "]"
        if (depth == 0) {
            sb.append(']').append(LF);
        } else {
            sb.append("],").append(LF);
        }
    }

    private void append(StringBuilder sb, Number[] values, int maxColumns) {
        if (values.length == 0) {
            return;
        }
        long limit = Math.min(values.length, maxColumns);
        sb.append(format(values[0]));
        for (int i = 1; i < limit; ++i) {
            sb.append(", ");
            sb.append(format(values[i]));
        }

        long remaining = values.length - limit;
        if (remaining > 0) {
            sb.append(", ... ").append(remaining).append(" more");
        }
    }

    private static final class FloatFormat extends NDFormat {

        private boolean exponential;
        private int precision;
        private int totalLength;

        public FloatFormat(NDArray array) {
            Number[] values = array.toArray();
            int maxIntPartLen = 0;
            int maxFractionLen = 0;
            int expFractionLen = 0;
            int maxExpSize = 2;
            boolean sign = false;

            double max = 0;
            double min = Double.MAX_VALUE;
            for (Number n : values) {
                double v = n.doubleValue();
                if (v < 0) {
                    sign = true;
                }

                if (!Double.isFinite(v)) {
                    int intPartLen = v < 0 ? 4 : 3;
                    if (totalLength < intPartLen) {
                        totalLength = intPartLen;
                    }
                    continue;
                }
                double abs = Math.abs(v);
                String str = String.format(Locale.ENGLISH, "%16e", abs);
                Matcher m = PATTERN.matcher(str);
                if (!m.matches()) {
                    throw new AssertionError("Invalid decimal value: " + str);
                }
                int fractionLen = m.group(1).length();
                if (expFractionLen < fractionLen) {
                    expFractionLen = fractionLen;
                }
                int expSize = m.group(2).length();
                if (expSize > maxExpSize) {
                    maxExpSize = expSize;
                }

                if (abs >= 1) {
                    int intPartLen = (int) Math.log10(abs) + 1;
                    if (v < 0) {
                        ++intPartLen;
                    }
                    if (intPartLen > maxIntPartLen) {
                        maxIntPartLen = intPartLen;
                    }
                    int fullFractionLen = fractionLen + 1 - intPartLen;
                    if (maxFractionLen < fullFractionLen) {
                        maxFractionLen = fullFractionLen;
                    }
                } else {
                    int intPartLen = v < 0 ? 2 : 1;
                    if (intPartLen > maxIntPartLen) {
                        maxIntPartLen = intPartLen;
                    }

                    int fullFractionLen = fractionLen + Integer.parseInt(m.group(2));
                    if (maxFractionLen < fullFractionLen) {
                        maxFractionLen = fullFractionLen;
                    }
                }

                if (abs > max) {
                    max = abs;
                }
                if (abs < min && abs > 0) {
                    min = abs;
                }
            }
            double ratio = max / min;
            if (max > 1.e8 || min < 0.0001 || ratio > 1000.) {
                exponential = true;
                precision = Math.min(PRECISION, expFractionLen);
                totalLength = precision + 4;
                if (sign) {
                    ++totalLength;
                }
            } else {
                precision = Math.min(4, maxFractionLen);
                int len = maxIntPartLen + precision + 1;
                if (totalLength < len) {
                    totalLength = len;
                }
            }
        }

        /** {@inheritDoc} */
        @Override
        public CharSequence format(Number value) {
            double d = value.doubleValue();
            if (Double.isNaN(d)) {
                return String.format(Locale.ENGLISH, "%" + totalLength + "s", "nan");
            } else if (Double.isInfinite(d)) {
                if (d > 0) {
                    return String.format(Locale.ENGLISH, "%" + totalLength + "s", "inf");
                } else {
                    return String.format(Locale.ENGLISH, "%" + totalLength + "s", "-inf");
                }
            }
            if (exponential) {
                precision = Math.max(PRECISION, precision);
                return String.format(Locale.ENGLISH, "% ." + precision + "e", value.doubleValue());
            }
            if (precision == 0) {
                String fmt = "%" + (totalLength - 1) + '.' + precision + "f.";
                return String.format(Locale.ENGLISH, fmt, value.doubleValue());
            }

            String fmt = "%" + totalLength + '.' + precision + 'f';
            String ret = String.format(Locale.ENGLISH, fmt, value.doubleValue());
            // Replace trailing zeros with space
            char[] chars = ret.toCharArray();
            for (int i = chars.length - 1; i >= 0; --i) {
                if (chars[i] == '0') {
                    chars[i] = ' ';
                } else {
                    break;
                }
            }
            return new String(chars);
        }
    }

    private static final class HexFormat extends NDFormat {

        /** {@inheritDoc} */
        @Override
        public CharSequence format(Number value) {
            return String.format(Locale.ENGLISH, "0x%02X", value.byteValue());
        }
    }

    private static final class IntFormat extends NDFormat {

        private boolean exponential;
        private int precision;
        private int totalLength;

        public IntFormat(NDArray array) {
            Number[] values = array.toArray();
            // scalar case
            if (values.length == 1) {
                totalLength = 1;
                return;
            }
            long max = 0;
            long negativeMax = 0;
            for (Number n : values) {
                long v = n.longValue();
                long abs = Math.abs(v);
                if (v < 0 && abs > negativeMax) {
                    negativeMax = abs;
                }
                if (abs > max) {
                    max = abs;
                }
            }

            if (max >= 1.e8) {
                exponential = true;
                precision = Math.min(PRECISION, (int) Math.log10(max) + 1);
            } else {
                int size = (max != 0) ? (int) Math.log10(max) + 1 : 1;
                int negativeSize = (negativeMax != 0) ? (int) Math.log10(negativeMax) + 2 : 2;
                totalLength = Math.max(size, negativeSize);
            }
        }

        /** {@inheritDoc} */
        @Override
        public CharSequence format(Number value) {
            if (exponential) {
                return String.format(Locale.ENGLISH, "% ." + precision + "e", value.floatValue());
            }
            return String.format(Locale.ENGLISH, "%" + totalLength + "d", value.longValue());
        }
    }

    private static final class BooleanFormat extends NDFormat {

        /** {@inheritDoc} */
        @Override
        public CharSequence format(Number value) {
            return value.byteValue() != 0 ? " true" : "false";
        }
    }
}
