using System;
using System.Collections.Generic;
using System.Text;
using MxNet.Numpy;
using NUnit.Framework;

namespace MxNet.Tests
{
    public class numpy_functions_tests
    {
        private ndarray a;
        private ndarray b;
        private ndarray c;

        [SetUp]
        public void Setup()
        {
            a = np.full(new Shape(3, 3), 3);
            b = np.full(new Shape(3, 3), 2);
            c = np.full(new Shape(3, 3), 5);
        }

        [Test]
        public void empty_test()
        {
            var x = np.empty(new Shape(3, 3));
            Assert.AreNotEqual(x, null);
            Assert.Pass();
        }

        [Test]
        public void array_test()
        {
            var x = np.array(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            Assert.AreNotEqual(x[2], 3);
            Assert.Pass();
        }
    }
}
