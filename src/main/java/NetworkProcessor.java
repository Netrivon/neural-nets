import java.io.File;
import java.util.List;

import com.thoughtworks.xstream.XStream;


public class NetworkProcessor {
	
	public static void processNetwork(String networkPath, String trainingSetPath) throws Exception {
		XStream xstream = new XStream();
		
		File networkFile = new File(networkPath);
		File trainingSetFile = new File(trainingSetPath);
	
		System.out.println("Processing Network...");
		System.out.println("networkFile: " + networkFile.getAbsolutePath());
		System.out.println("trainingSetFile: " + trainingSetFile.getAbsolutePath());
		
		Network network = AdalineUtils.readNetwork(networkFile);
		
		System.out.println("input network: " + network);
		System.out.println(xstream.toXML(network));
		
		TrainingSet trainingSet = AdalineUtils.readTrainingSet(trainingSetFile);
		
		network.train(trainingSet);
		
		System.out.println("weights after training: " + network.getWeightMatrix());
		
		List<Double> input = null;
		List<Double> output = null;
		
		System.out.println("trained network outputs: ");
		for(TrainingSample trainingSample : trainingSet.getTrainingSetSamples()) {
			input = trainingSample.getInputVector();
			output = network.process(input);
			
			System.out.println("input:	" + input);
			System.out.println("output:	" + output);
		}
		
		System.out.println("output network: " + network);
		System.out.println(xstream.toXML(network));
	}
	
	public static void main(String [] args) throws Exception {
		
		if(args.length != 2) {
			System.out.println("Usage: NetworkProcessor <networkPath> <trainingSetPath>");
			System.out.println("<networkPath>: Network definition path (eg. adalien-network.xml)");
			System.out.println("<trainingSetPath>: Network training set path (eg. and-training-set.xml)");
		}
		
		String networkPath = args[0];
		String trainingSetPath = args[1];
		
		processNetwork(networkPath, trainingSetPath);
		
	}
}
