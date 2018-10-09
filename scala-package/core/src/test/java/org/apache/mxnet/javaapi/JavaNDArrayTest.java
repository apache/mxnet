package org.apache.mxnet.javaapi;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertTrue;

public class JavaNDArrayTest {
    @Test
    public void testCreateNDArray() {
        NDArray nd = new NDArray(new float[]{1.0f, 2.0f, 3.0f},
                new Shape(new int[]{1, 3}),
                new Context("cpu", 0));
        int[] arr = new int[]{1, 3};
        assertTrue(Arrays.equals(nd.shape().toArray(), arr));
        assertTrue(nd.at(0).at(0).toArray()[0] == 1.0f);
        List<Float> list = new ArrayList<Float>();
        list.add(1.0f);
        list.add(2.0f);
        list.add(3.0f);
        nd.dispose();
        // Second way creating NDArray
        nd = NDArray.array(list,
                new Shape(new int[]{1, 3}),
                new Context("cpu", 0));
        assertTrue(Arrays.equals(nd.shape().toArray(), arr));
    }

    @Test
    public void testZeroOneEmpty(){
        NDArray ones = NDArray.ones(new Context("cpu", 0), new int[]{100, 100});
        NDArray zeros = NDArray.zeros(new Context("cpu", 0), new int[]{100, 100});
        NDArray empty = NDArray.zeros(new Context("cpu", 0), new int[]{100, 100});
        int[] arr = new int[]{100, 100};
        assertTrue(Arrays.equals(ones.shape().toArray(), arr));
        assertTrue(Arrays.equals(zeros.shape().toArray(), arr));
        assertTrue(Arrays.equals(empty.shape().toArray(), arr));
    }

    @Test
    public void testComparison(){
        NDArray nd = new NDArray(new float[]{1.0f, 2.0f, 3.0f}, new Shape(new int[]{3}), new Context("cpu", 0));
        NDArray nd2 = new NDArray(new float[]{3.0f, 4.0f, 5.0f}, new Shape(new int[]{3}), new Context("cpu", 0));
        nd = nd.add(nd2);
        float[] greater = new float[]{1, 1, 1};
        assertTrue(Arrays.equals(nd.greater(nd2).toArray(), greater));
        nd = nd.subtract(nd2);
        nd = nd.subtract(nd2);
        float[] lesser = new float[]{0, 0, 0};
        assertTrue(Arrays.equals(nd.greater(nd2).toArray(), lesser));
    }

    @Test
    public void testGenerated(){
        NDArray$ NDArray = NDArray$.MODULE$;
        float[] arr = new float[]{1.0f, 2.0f, 3.0f};
        NDArray nd = new NDArray(arr, new Shape(new int[]{3}), new Context("cpu", 0));
        float result = NDArray.norm(nd).invoke().get().toArray()[0];
        float cal = 0.0f;
        for (float ele : arr) {
            cal += ele * ele;
        }
        cal = (float) Math.sqrt(cal);
        assertTrue(Math.abs(result - cal) < 1e-5);
        NDArray dotResult = new NDArray(new float[]{0}, new Shape(new int[]{1}), new Context("cpu", 0));
        NDArray.dot(nd, nd).setout(dotResult).invoke().get();
        assertTrue(Arrays.equals(dotResult.toArray(), new float[]{14.0f}));
    }
}
