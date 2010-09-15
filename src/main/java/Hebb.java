
public class Hebb extends Adaline implements Network {
	@Override
	public void train(TrainingSet trainingSet) {
		initWeights();
		
		for(int j = 0; j < getNumOutputElements(); j++) {
			Double desiredOutput = 0.0;
			Double actualOutput = 0.0;
			
			long iter = 0;
			
			for(TrainingSample trainingSample : trainingSet.getTrainingSetSamples()) {
				
				desiredOutput = trainingSample.getOutputVector().get(j);
				actualOutput = process(trainingSample.getInputVector()).get(j);
				
				iter = 0;
				
				while(Math.pow(desiredOutput - actualOutput, 2) > getMinError() && iter < getMaxIterations()) {
					
					double [] weights = getWeightMatrix().getColumn(j);
					
					// generalized hebbian weight algorithm
					for(int i = 0; i < weights.length; i++) {
						
						double sum = 0;
						
						double [] subweights = getWeightMatrix().getRow(i);
						
						for(int k = 0; k < j; k++) {
							sum += subweights[k]*trainingSample.getOutputVector().get(k);
						}
						
						double deltaWeight = getLearningRate()*(actualOutput*trainingSample.getInputVector().get(i) - actualOutput*sum);
						getWeightMatrix().setEntry(i, j, weights[i] + deltaWeight);
					}
					
					desiredOutput = trainingSample.getOutputVector().get(j);
					actualOutput = process(trainingSample.getInputVector()).get(j);
					iter++;
				}
				
			}
		}
	}
}
