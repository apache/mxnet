package org.apache.mxnet.training.initializer;

import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;

/**
 * {@code XavierInitializer} is an {@link Initializer} that performs "Xavier" initialization for
 * parameters. This initializer is designed to keep the scale of gradients roughly the same in all
 * layers. It was originally defined in the paper <a
 * href="http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"><i>Understanding the difficulty of
 * training deep feedforward neural networks</i></a>.
 *
 * <p>{@code XavierInitializer} is specified by the type of random distribution({@link RandomType}),
 * the factor type({@link FactorType}), and the magnitude of the scale. By default, {@link
 * RandomType} is {@code UNIFORM} and {@link FactorType} is {@code AVG}. The initializer fills the
 * weights with random numbers in the range of \([-c, c]\), where \(c = \sqrt{\frac{3.}{0.5 *
 * (n_{in} + n_{out})}}\) where \(n_{in}\) is the number of neurons feeding into weights, and
 * \(n_{out}\) is the number of neurons the result is fed to.
 *
 * <p>If {@link RandomType} is {@code UNIFORM} and {@link FactorType} is {@code IN}, then \(c =
 * \sqrt{\frac{3.}{n_{in}}}\). Similarly when {@link FactorType} is {@code OUT}, then \(c =
 * \sqrt{\frac{3.}{n_{out}}}\).
 *
 * <p>If {@link RandomType} is {@code GAUSSIAN} and {@link FactorType} is {@code AVG}, the
 * initializer fills the weights with numbers from normal distribution with a standard deviation of
 * \(\sqrt{\frac{3.}{0.5 * (n_{in} + n_{out})}}\).
 *
 * <p>Another common setting of the {@code XavierInitializer} is defined in the paper <a
 * href="https://arxiv.org/abs/1502.01852"><i>Delving Deep into Rectifiers: Surpassing Human-Level
 * Performance on ImageNet Classification</i></a>. These settings better handle non-linearity when
 * preserving the variance across layers in a neural network. It can be initialized with {@code new
 * XavierInitializer(RandomType.GAUSSIAN, FactorType.IN, 2))}.
 */
public class XavierInitializer implements Initializer {

    /** Enum for different types of random distributions. */
    public enum RandomType {
        UNIFORM,
        GAUSSIAN
    }

    /** Enum for different types of factor type. */
    public enum FactorType {
        AVG,
        IN,
        OUT
    }

    private RandomType randomType;
    private FactorType factorType;
    private float magnitude;

    /**
     * Initializes a Xavier initializer.
     *
     * @param randomType the random generator type, can be GAUSSIAN or UNIFORM
     * @param factorType the factor type, can be one of AVG, IN, or OUT
     * @param magnitude the scale of the random number
     */
    public XavierInitializer(RandomType randomType, FactorType factorType, float magnitude) {
        this.randomType = randomType;
        this.factorType = factorType;
        this.magnitude = magnitude;
    }

    /** Creates a new instance of {@code XavierInitializer}. */
    public XavierInitializer() {
        this(RandomType.UNIFORM, FactorType.AVG, 6f);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray initialize(MxResource parent, Shape shape, DataType dataType, Device device) {

        float hwScale;
        long dimension = shape.dimension();
        if (dimension < 2) {
            throw new IllegalArgumentException(
                    "XavierInitializer cannot be applied to Shape with dimension: "
                            + dimension
                            + ", it requires shape to be at least 2D.");
        } else if (dimension == 2) {
            hwScale = 1.0f;
        } else {
            Shape shapeSliced = shape.slice(2);
            hwScale = shapeSliced.size();
        }
        float fanIn = shape.get(1) * hwScale;
        float fanOut = shape.head() * hwScale;
        float factor;
        switch (factorType) {
            case AVG:
                factor = (fanIn + fanOut) / 2.0f;
                break;
            case IN:
                factor = fanIn;
                break;
            case OUT:
                factor = fanOut;
                break;
            default:
                throw new IllegalArgumentException(
                        "Invalid factor type, valid types are: avg, in, out");
        }
        if (factor == 0f) {
            throw new IllegalStateException(
                    "Xavier initializer factor is 0, please check your input shape.");
        }
        float scale = (float) StrictMath.sqrt(magnitude / factor);

        switch (randomType) {
            case UNIFORM:
                return MxNDArray.randomUniform(parent, -scale, scale, shape, dataType, device);
            case GAUSSIAN:
                return MxNDArray.randomNormal(parent, 0f, scale, shape, dataType, device);
            default:
                throw new IllegalArgumentException("Invalid randomType");
        }
    }
}