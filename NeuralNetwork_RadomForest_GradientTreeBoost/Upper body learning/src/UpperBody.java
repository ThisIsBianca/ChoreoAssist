

//TRAINING AND TESTING FOR UPPER BODY WITHOUT NEGATIVES



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

		//exclude lines which end with "0.0", that is, negatives
		for (int i = 0; i < linesTrain.length; i++) {
			String j = linesTrain[i];
			if (!j.endsWith("0.0")) {
				lister.add(j);
			}
		}
		for (int i = 0; i < linesTest.length; i++) {
			String j = linesTest[i];
			if (!j.endsWith("0.0")) {
				lister2.add(j);
			}
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
			
//			System.out.println(index + ": " + components[1] + "," + components[2] + "," + components[3]
//			           + "," + components[4] + "," + components[5]);
			

			// if (PApplet.parseInt(components[5]) == 0) {
			// continue;
			// }

			data[index][0] = PApplet.map(PApplet.parseFloat(components[1]), 0, 100, 0, 1);
			data[index][1] = PApplet.map(PApplet.parseFloat(components[2]), 0, 100, 0, 1);
			data[index][2] = PApplet.map(PApplet.parseFloat(components[3]), 0, 100, 0, 1);
			data[index][3] = PApplet.map(PApplet.parseFloat(components[4]), 0, 100, 0, 1);
			labels[index] = PApplet.parseInt(components[5]) - 1;
			
			
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
			
//			System.out.println(index + ": " + components[1] + "," + components[2] + "," + components[3]
//			           + "," + components[4] + "," + components[5]);
			

			// if (PApplet.parseInt(components[5]) == 0) {
			// continue;
			// }

			testData[index2][0] = PApplet.map(PApplet.parseFloat(components[1]), 0, 100, 0, 1);
			testData[index2][1] = PApplet.map(PApplet.parseFloat(components[2]), 0, 100, 0, 1);
			testData[index2][2] = PApplet.map(PApplet.parseFloat(components[3]), 0, 100, 0, 1);
			testData[index2][3] = PApplet.map(PApplet.parseFloat(components[4]), 0, 100, 0, 1);
			labels2[index2] = PApplet.parseInt(components[5]) - 1;

			
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
		
//		double[][] testData = {{28.740711,85.328186,29.914495,89.43523},
//								{42.44326,70.27071,50.913303,42.906223},
//								{36.541042,72.08018,30.647558,80.99253},
//				                {18.848885,4.6686673,19.124495,5.8006673},
//				                {13.327746,8.364914,16.129618,9.452747},
//				                {7.1946683,10.423285,2.8939075,11.210916},
//				                {14.656629,2.5793216,55.030365,45.457664},
//				                {13.616934,0.6205026,56.20149,53.312992},
//				                {14.260979,12.13867,54.79631,49.043503}};


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
		
		
		//testing
		   {
		   
		   int[] predNet = new int[testData.length];
		   for (int i = 0; i < testData.length; i++) {
				predNet[i] = net.predict(testData[i]);
				System.out.println("The Neural Network says: " + (predNet[i] + 1));
			}
		   double testError = error(labels2, predNet);
		   System.out.format("NeuralNetwork -- testing error = %.2f%%\n", 100 * testError);

		   }
		   
		   {
		   
		   int[] predRdf = new int[testData.length];
		   for (int i = 0; i < testData.length; i++) {
				predRdf[i] = rdf.predict(testData[i]);
				System.out.println("The Random Forest says: " + (predRdf[i]+1));
			}
		   double testError = error(labels2, predRdf);
		   System.out.format("RandomForest -- testing error = %.2f%%\n", 100 * testError);

		   }
		   
		   {
		   
		   int[] predGtb = new int[testData.length];
		   for (int i = 0; i < testData.length; i++) {
				predGtb[i] = gtb.predict(testData[i]);
				System.out.println("The Gradient Tree says: " + (predGtb[i]+1));
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
