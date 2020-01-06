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

package org.apache.mxnet.javaapi;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.Test;
import java.util.ArrayList;
import java.util.Arrays;

public class ShapeTest {
	@Test
	public void testArrayConstructor()
	{
		new Shape(new int[] {3, 4, 5});
	}
	
	@Test
	public void testListConstructor()
	{
		ArrayList<Integer> arrList = new ArrayList<Integer>();
		arrList.add(3);
		arrList.add(4);
		arrList.add(5);
		new Shape(arrList);
	}
	
	@Test
	public void testApply()
	{
		Shape jS = new Shape(new int[] {3, 4, 5});
		assertEquals(jS.apply(1), 4);
	}
	
	@Test
	public void testGet()
	{
		Shape jS = new Shape(new int[] {3, 4, 5});
		assertEquals(jS.get(1), 4);
	}
	
	@Test
	public void testSize()
	{
		Shape jS = new Shape(new int[] {3, 4, 5});
		assertEquals(jS.size(), 3);
	}
	
	@Test
	public void testLength()
	{
		Shape jS = new Shape(new int[] {3, 4, 5});
		assertEquals(jS.length(), 3);
	}
	
	@Test
	public void testDrop()
	{
		Shape jS = new Shape(new int[] {3, 4, 5});
		ArrayList<Integer> l = new ArrayList<Integer>();
		l.add(4);
		l.add(5);
		assertTrue(jS.drop(1).toVector().equals(l));
	}
	
	@Test
	public void testSlice()
	{
		Shape jS = new Shape(new int[] {3, 4, 5});
		ArrayList<Integer> l = new ArrayList<Integer>();
		l.add(4);
		assertTrue(jS.slice(1,2).toVector().equals(l));
	}
	
	@Test
	public void testProduct()
	{
		Shape jS = new Shape(new int[] {3, 4, 5});
		assertEquals(jS.product(), 60);
	}
	
	@Test
	public void testHead()
	{
		Shape jS = new Shape(new int[] {3, 4, 5});
		assertEquals(jS.head(), 3);
	}
	
	@Test
	public void testToArray()
	{
		Shape jS = new Shape(new int[] {3, 4, 5});
		assertTrue(Arrays.equals(jS.toArray(), new int[] {3,4,5}));
	}
	
	@Test
	public void testToVector()
	{
		Shape jS = new Shape(new int[] {3, 4, 5});
		ArrayList<Integer> l = new ArrayList<Integer>();
		l.add(3);
		l.add(4);
		l.add(5);
		assertTrue(jS.toVector().equals(l));
	}
}
