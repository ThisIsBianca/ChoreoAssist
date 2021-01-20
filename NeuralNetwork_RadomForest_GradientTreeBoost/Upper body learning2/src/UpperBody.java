
//TRAINING AND TESTING FOR UPPER BODY WITH NEGATIVES


import java.io.File;
import java.util.LinkedList;
import java.util.List;

import processing.core.PApplet;
import smile.classification.GradientTreeBoost;
import smile.classification.NeuralNetwork;
import smile.classification.RandomForest;

public class UpperBody {
	public static void main(String[] args) {
		String[] linesTrain = PApplet.loadStrings(new File("C:\\Users\\s147181\\Documents\\master year II\\intelligence in interaction\\coordinates of limbs - second recording\\upper_body_learning\\data\\for learning\\Everybody_upper_body_second_train.csv"));
		String[] linesTest = PApplet.loadStrings(new File("C:\\Users\\s147181\\Documents\\master year II\\intelligence in interaction\\coordinates of limbs - second recording\\upper_body_learning\\data\\for learning\\Everybody_upper_body_second_test.csv"));

		List<String> lister = new LinkedList<String>();
		List<String> lister2 = new LinkedList<String>();


		for (int i = 0; i < linesTrain.length; i++) {
			String j = linesTrain[i];
//			if (!j.endsWith("0.0")) {
				lister.add(j);
//			}
		}
		for (int i = 0; i < linesTest.length; i++) {
			String j = linesTest[i];
//			if (!j.endsWith("0.0")) {
				lister2.add(j);
//			}
		}
		
		//data set and label set for training
		double[][] data = new double[lister.size()][4];
		int[] labels = new int[lister.size()];
		
		//data set and label set for testing
		double[][] testData = new double[lister2.size()][4];
		int[] labels2 = new int[lister2.size()];

		//extract data set for learning
		int index = 0;
		for (String line : lister) {
			String[] components = line.split(",");

			if (components.length != 6) {
				System.out.println(line);
				System.out.println("EXIT");
				System.exit(0);
			} 
			

			data[index][0] = PApplet.map(PApplet.parseFloat(components[1]), 0, 100, 0, 1);
			data[index][1] = PApplet.map(PApplet.parseFloat(components[2]), 0, 100, 0, 1);
			data[index][2] = PApplet.map(PApplet.parseFloat(components[3]), 0, 100, 0, 1);
			data[index][3] = PApplet.map(PApplet.parseFloat(components[4]), 0, 100, 0, 1);
			labels[index] = PApplet.parseInt(components[5]);
			
			
			//System.out.println(labels[index]);
            

			index++;
		}
		
		
		//extract data set for testing
		int index2 = 0;
		for (String line : lister2) {
			String[] components = line.split(",");

			if (components.length != 6) {
				System.out.println(line);
				System.out.println("EXIT");
				System.exit(0);
			} 
			

			testData[index2][0] = PApplet.map(PApplet.parseFloat(components[1]), 0, 100, 0, 1);
			testData[index2][1] = PApplet.map(PApplet.parseFloat(components[2]), 0, 100, 0, 1);
			testData[index2][2] = PApplet.map(PApplet.parseFloat(components[3]), 0, 100, 0, 1);
			testData[index2][3] = PApplet.map(PApplet.parseFloat(components[4]), 0, 100, 0, 1);
			labels2[index2] = PApplet.parseInt(components[5]);

			
			//			System.out.println(testData[index2][0] + ", " + testData[index2][1] +
//					 ", " + testData[index2][2]+ ", " + testData[index2][3]);
			
			
			//System.out.println(labels[index]);
            

			index2++;
		}


		// training
		int units = 50;
		// NeuralNetwork net = new NeuralNetwork(NeuralNetwork.ErrorFunction.CROSS_ENTROPY,
		// NeuralNetwork.ActivationFunction.SOFTMAX, 4, units, 4);
		NeuralNetwork net = new NeuralNetwork(NeuralNetwork.ErrorFunction.CROSS_ENTROPY,
				NeuralNetwork.ActivationFunction.SOFTMAX, 4, units, 4);

		for (int i = 0; i < 1000; i++) {
			net.learn(data, labels);
			// System.out.println(i);
		}

		int trees = 25;
		RandomForest rdf = new RandomForest(data, labels, trees);

		GradientTreeBoost gtb = new GradientTreeBoost(data, labels, trees);
		


		// prediction
		{
			int[] pred = new int[labels.length];
			for (int i = 0; i < labels.length; i++) {
				pred[i] = net.predict(data[i]);
			}
			double trainError = error(labels, pred);
			System.out.format("NeuralNetwork -- training error = %.2f%%\n", 100 * trainError);
		}

		{
			int[] pred = new int[labels.length];
			for (int i = 0; i < labels.length; i++) {
				pred[i] = rdf.predict(data[i]);
			}
			double trainError = error(labels, pred);
			System.out.format("RandomForest -- training error = %.2f%%\n", 100 * trainError);
		}

		{
			int[] pred = new int[labels.length];
			for (int i = 0; i < labels.length; i++) {
				pred[i] = gtb.predict(data[i]);
				//System.out.println(pred[i]);
			}
			double trainError = error(labels, pred);
			System.out.format("GradientTreeBoost -- training error = %.2f%%\n", 100 * trainError);
		}
		
		
		//testing, print out the misclassified ones
		   {
		   int errors =0;
		   int[] predNet = new int[testData.length];
		   for (int i = 0; i < testData.length; i++) {
				predNet[i] = net.predict(testData[i]);
				if(labels2[i] != predNet[i]) {
				System.out.println("For row " + i + " NeuralNetwork says: " 
						+ predNet[i] + " but it is " + labels2[i]);
				errors ++;
				}
				//System.out.println("The Neural Network says: " + (predNet[i]));
			}
		   double testError = error(labels2, predNet);
		   System.out.println(errors + " " + testData.length);
		   System.out.format("NeuralNetwork -- testing error = %.2f%%\n", 100 * testError);

		   }
		   
		   {
		   
		   int[] predRdf = new int[testData.length];
		   for (int i = 0; i < testData.length; i++) {
				predRdf[i] = rdf.predict(testData[i]);
				if(labels2[i] != predRdf[i]) {
				System.out.println("For row " + i + " RandomForrest says: " 
						+ predRdf[i] + " but it is " + labels2[i]);
				}
				//System.out.println("The Random Forest says: " + (predRdf[i]));
			}
		   double testError = error(labels2, predRdf);
		   System.out.format("RandomForest -- testing error = %.2f%%\n", 100 * testError);

		   }
		   
		   {
		   
		   int[] predGtb = new int[testData.length];
		   for (int i = 0; i < testData.length; i++) {
				predGtb[i] = gtb.predict(testData[i]);
				if(labels2[i] != predGtb[i]) {
				System.out.println("For row " + i + " GradientTreeBoost says: " 
						+ predGtb[i] + " but it is " + labels2[i]);
				}
				//System.out.println("The Gradient Tree says: " + (predGtb[i]));
			}
		   double testError = error(labels2, predGtb);
		   System.out.format("GradientTreeBoost -- testing error = %.2f%%\n", 100 * testError);

		   }

	}

	/**
	 * Returns the error rate.
	 */
	static double error(int[] x, int[] y) {
		int e = 0;

		for (int i = 0; i < x.length; i++) {
			if (x[i] != y[i]) {
				e++;
			}
		}

		return (double) e / x.length;
	}
	
}
