import java.io.File;

import junit.framework.Assert;

import org.junit.Test;


public class MadalineTest {
	
	@Test
	public void testReadMadaline() throws Exception {
		
		File networkFile = new File("src/test/resources/madaline-network.xml");
		
		MultiLayerNetwork madaline = (MultiLayerNetwork) AdalineUtils.readNetwork(networkFile);
		
		Assert.assertTrue("El numero de capas debe ser 2", madaline.getNumLayers() == 2);
	}
	
	
	@Test
	public void testNetworkTrain() throws Exception {
		
		String networkPath = "src/test/resources/madaline-network.xml";
		String trainingSetPath = "src/test/resources/and-training-set.xml";
		NetworkProcessor.processNetwork(networkPath, trainingSetPath);
		
	}
	
}
