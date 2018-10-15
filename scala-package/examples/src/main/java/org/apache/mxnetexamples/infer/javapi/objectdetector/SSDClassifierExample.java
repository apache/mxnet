package org.apache.mxnetexamples.infer.javapi.objectdetector;

import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.infer.javaapi.ImageClassifier;
import org.apache.mxnet.infer.javaapi.ObjectDetector;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javafx.util.Pair;

import java.io.File;

public class SSDClassifierExample {
	@Option(name = "--model-path-prefix", usage = "the input model directory and prefix of the model")
	private String modelPathPrefix = "/model/ssd_resnet50_512";
	@Option(name = "--input-image", usage = "the input image")
	private String inputImagePath = "/images/dog.jpg";
	@Option(name = "--input-dir", usage = "the input batch of images directory")
	private String inputImageDir = "/images/";
	
	final static Logger logger = LoggerFactory.getLogger(SSDClassifierExample.class);
	
	static List<List<Pair<String, List<Float>>>> runObjectDetectionSingle(String modelPathPrefix, String inputImagePath, List<Context> context) {
		Shape inputShape = new Shape(new int[] {1, 3, 512, 512});
		List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
		inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
		BufferedImage img = ImageClassifier.loadImageFromFile(inputImagePath);
		ObjectDetector objDetector = new ObjectDetector(modelPathPrefix, inputDescriptors, context, 0);
		return objDetector.imageObjectDetect(img, 3);
	}
	
	static List<List<List<Pair<String, List<Float>>>>> runObjectDetectionBatch(String modelPathPrefix, String inputImageDir, List<Context> context) {
		Shape inputShape = new Shape(new int[]{1, 3, 512, 512});
		List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
		inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
		ObjectDetector objDetector = new ObjectDetector(modelPathPrefix, inputDescriptors, context, 0);
		
		// Loading batch of images from the directory path
		List<List<String>> batchFiles = generateBatches(inputImageDir, 20);
		List<List<List<Pair<String, List<Float>>>>> outputList = new ArrayList<List<List<Pair<String, List<Float>>>>>();
		
		for (List<String> batchFile : batchFiles) {
			List<BufferedImage> imgList = ImageClassifier.loadInputBatch(batchFile);
			// Running inference on batch of images loaded in previous step
			List<List<Pair<String, List<Float>>>> tmp = objDetector.imageBatchObjectDetect(imgList, 5);
			outputList.add(tmp);
		}
		return outputList;
	}
	
	static List<List<String>> generateBatches(String inputImageDirPath, int batchSize) {
		File dir = new File(inputImageDirPath);

		List<List<String>> output = new ArrayList<List<String>>();
		List<String> batch = new ArrayList<String>();
		for (File imgFile : dir.listFiles()) {
			batch.add(imgFile.getPath());
			if (batch.size() == batchSize) {
				output.add(batch);
				batch = new ArrayList<String>();
			}
		}
		if (batch.size() > 0) {
			output.add(batch);
		}
		return output;
	}
	
	public static void main(String[] args) {
		SSDClassifierExample inst = new SSDClassifierExample();
		CmdLineParser parser  = new CmdLineParser(inst);
		try {
			parser.parseArgument(args);
		} catch (Exception e) {
			logger.error(e.getMessage(), e);
			parser.printUsage(System.err);
			System.exit(1);
		}
		
		String mdprefixDir = inst.modelPathPrefix;
		String imgPath = inst.inputImagePath;
		String imgDir = inst.inputImageDir;
		
		if (!checkExist(Arrays.asList(mdprefixDir + "-symbol.json", imgDir, imgPath))) {
			logger.error("Model or input image path does not exist");
			System.exit(1);
		}
		
		List<Context> context = new ArrayList<Context>();
		if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
				Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
			context.add(Context.gpu());
		} else {
			context.add(Context.cpu());
		}
		
		try {
			Shape inputShape = new Shape(new int[] {1, 3, 512, 512});
			Shape outputShape = new Shape(new int[] {1, 6132, 6});
			
			
			int width = inputShape.get(2);
			int height = inputShape.get(3);
			String outputStr = "\n";
			
			List<List<Pair<String, List<Float>>>> output = runObjectDetectionSingle(mdprefixDir, imgPath, context);
			
			for (List<Pair<String, List<Float>>> ele : output) {
				for (Pair<String, List<Float>> i : ele) {
					outputStr += "Class: " + i.getKey() + "\n";
					List<Float> arr = i.getValue();
					outputStr += "Probabilties: " + arr.get(0) + "\n";
					
					List<Float> coord = Arrays.asList(arr.get(1) * width, arr.get(2) * height, arr.get(3) * width, arr.get(4) * height);
					StringBuilder sb = new StringBuilder();
					for (float c: coord) {
						sb.append(", ").append(c);
					}
					outputStr += "Coord:" + sb.substring(2)+ "\n";
				}
			}
			logger.info(outputStr);
			
			List<List<List<Pair<String, List<Float>>>>> outputList = runObjectDetectionBatch(mdprefixDir, imgDir, context);
			
			outputStr = "\n";
			int index = 0;
			for (List<List<Pair<String, List<Float>>>> i: outputList) {
				for (List<Pair<String, List<Float>>> j : i) {
					outputStr += "*** Image " + (index + 1) + "***" + "\n";
					for (Pair<String, List<Float>> k : j) {
						outputStr += "Class: " + k.getKey() + "\n";
						List<Float> arr = k.getValue();
						outputStr += "Probabilties: " + arr.get(0) + "\n";
						List<Float> coord = Arrays.asList(arr.get(1) * width, arr.get(2) * height, arr.get(3) * width, arr.get(4) * height);
						
						StringBuilder sb = new StringBuilder();
						for (float c : coord) {
							sb.append(", ").append(c);
						}
						outputStr += "Coord:" + sb.substring(2) + "\n";
					}
					index++;
				}
			}
			logger.info(outputStr);
			
		} catch (Exception e) {
			logger.error(e.getMessage(), e);
			parser.printUsage(System.err);
			System.exit(1);
		}
		System.exit(0);
	}
	
	static Boolean checkExist(List<String> arr)  {
		Boolean exist = true;
		for (String item : arr) {
			exist = new File(item).exists() && exist;
			if (!exist) {
				logger.error("Cannot find: " + item);
			}
		}
		return exist;
	}
}
