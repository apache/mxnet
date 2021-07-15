package org.apache.mxnet.ndarray.types;

import org.apache.mxnet.ndarray.MxNDArray;

import java.util.stream.IntStream;

/**
 * An enum to represent the meaning of a particular axis in an {@link MxNDArray}.
 *
 * <p>The options are:
 *
 * <ul>
 *   <li>{@link LayoutType#BATCH} - Different elements in a batch, usually from a {@link
 *       ai.djl.translate.StackBatchifier}.
 *   <li>{@link LayoutType#CHANNEL} - Each channel represents a different aspect of the data such as
 *       RGB showing different color channels.
 *   <li>{@link LayoutType#DEPTH} - The depth of a 3-D input
 *   <li>{@link LayoutType#HEIGHT} - The width of a multi-dimensional input, usually an image.
 *   <li>{@link LayoutType#WIDTH} - The height of a multi-dimensional input, usually an image.
 *   <li>{@link LayoutType#TIME} - The time within a sequence such as text or video.
 *   <li>{@link LayoutType#UNKNOWN} - A unknown or otherwise unrepresentable layout type.
 * </ul>
 */
public enum LayoutType {
    BATCH('N'),
    CHANNEL('C'),
    DEPTH('D'),
    HEIGHT('H'),
    WIDTH('W'),
    TIME('T'),
    UNKNOWN('?');

    private char value;

    LayoutType(char value) {
        this.value = value;
    }

    /**
     * Returns the character representation of the layout type.
     *
     * @return the character representation of the layout type
     */
    public char getValue() {
        return value;
    }

    /**
     * Converts the character to the matching layout type.
     *
     * @param value the character to convert
     * @return the matching layout type
     * @throws IllegalArgumentException thrown if the character does not match any layout type
     */
    public static LayoutType fromValue(char value) {
        for (LayoutType type : LayoutType.values()) {
            if (value == type.value) {
                return type;
            }
        }
        throw new IllegalArgumentException(
                "The value does not match any layoutTypes. Use '?' for Unknown");
    }

    /**
     * Converts each character to the matching layout type.
     *
     * @param layout the character string to convert
     * @return the list of layout types for each character in the string
     * @throws IllegalArgumentException thrown if the character does not match any layout type
     */
    public static LayoutType[] fromValue(String layout) {
        return IntStream.range(0, layout.length())
                .mapToObj(i -> fromValue(layout.charAt(i)))
                .toArray(LayoutType[]::new);
    }

    /**
     * Converts a layout type array to a string of the character representations.
     *
     * @param layouts the layout type to convert
     * @return the string of the character representations
     */
    public static String toString(LayoutType[] layouts) {
        StringBuilder sb = new StringBuilder(layouts.length);
        for (LayoutType layout : layouts) {
            sb.append(layout.getValue());
        }
        return sb.toString();
    }
}
