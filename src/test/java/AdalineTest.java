import java.io.File;
import java.util.List;

import junit.framework.Assert;

import org.junit.Test;

import com.thoughtworks.xstream.XStream;

public class AdalineTest {
	@Test
	public void testReadAndNetworkFile() throws Exception {

		File networkFile = new File("src/test/resources/adaline-network.xml");

		Network network = AdalineUtils.readNetwork(networkFile);

		Assert.assertTrue("Number of input elements must be 2", network
				.getNumInputElements() == 2);
		Assert.assertTrue("Number of output elements must be 1", network
				.getNumOutputElements() == 1);
		
	}

	@Test
	public void testReadAndTrainingSet() throws Exception {
		File trainingSetFile = new File(
				"src/test/resources/and-training-set.xml");

		TrainingSet trainingSet = AdalineUtils.readTrainingSet(trainingSetFile);
		Assert.assertTrue("Training set cannot be null", trainingSet != null);
		Assert.assertTrue("Training set samples cannot be null", trainingSet
				.getTrainingSetSamples() != null);
		Assert.assertTrue("Training set samples size must be 4", trainingSet
				.getTrainingSetSamples().size() == 4);

		for (TrainingSample trainingSample : trainingSet
				.getTrainingSetSamples()) {
			Assert.assertTrue("Training sample cannot be null",
					trainingSample != null);

			System.out.println("trainingSample input vector: \t"
					+ trainingSample.getInputVector());

			Assert.assertTrue("Training sample input vector cannot be null",
					trainingSample.getInputVector() != null);
			Assert.assertTrue("Training sample input vector size must be 2",
					trainingSample.getInputVector().size() == 2);

			System.out.println("trainingSample output vector: \t"
					+ trainingSample.getOutputVector());

			Assert.assertTrue("Training sample output vector cannot be null",
					trainingSample.getOutputVector() != null);
			Assert.assertTrue("Training sample output vector size must be 1",
					trainingSample.getOutputVector().size() == 1);

		}

	}
	
	
	@Test
	public void testNetworkTrain() throws Exception {
		
		String networkPath = "src/test/resources/adaline-network.xml";
		String trainingSetPath = "src/test/resources/and-training-set.xml";
		NetworkProcessor.processNetwork(networkPath, trainingSetPath);
		
	}
}
