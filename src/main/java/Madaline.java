import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;


public class Madaline implements MultiLayerNetwork {
	private int numInputElements;
	private int numOutputElements;
	private int numLayers;
	
	private double learningRate;
	private double minError;
	private double bias;
	private long maxIterations;

	private List<RealMatrix> weightMatrices = new ArrayList<RealMatrix>();
	
	private List<Double> outputElements = new ArrayList<Double>();
	
	
	
	// assume the number of elements in each hidden layer will
	// be equal to the number of input elements
	protected void initWeights() {
		
		weightMatrices = new ArrayList<RealMatrix>();
		
		for(int k = 0; k < numLayers; k++) {
			
			double [][] d = null;
			
			if(k == numLayers - 1) {
				// last layer ?
				d = new double[numInputElements][numOutputElements];
			} else {
				d = new double[numInputElements][numInputElements];
			}
			
			
			for(int i = 0; i < numInputElements; i++) {
				for(int j = 0; j < numOutputElements; j++) {
					d[i][j] = Math.random();
				}
			}
			
			RealMatrix weightMatrix = new Array2DRowRealMatrix(d);
			
			weightMatrices.add(weightMatrix);
			
		}
		
	}
	
	
	private Double sigmoid(Double value) {
		return 1.0 / (1 + Math.exp(-1000 * value));
	}
	
	
	public List<Double> processSingle(List<Double> inputElements, RealMatrix weightMatrix, int numOutputElements) {
		
		List<Double> outputList = new ArrayList<Double>();
		
		for(int j = 0; j < numOutputElements; j++) {
			Double output = 0.0;
			
			for(int i = 0; i < inputElements.size(); i++) {
				output += inputElements.get(i) * weightMatrix.getEntry(i, j);
			}
			
			output += bias;
			
			output = sigmoid(output);
			
			outputList.add(output);
		}
		
		return outputList;
	}
	
	
	public List<Double> process(List<Double> inputElements) {
		
		List<Double> outputList = null;
		
		for(int k = 0; k < numLayers; k++) {
			if(k == numLayers - 1) {
				// last hidden layer ?
				outputList = processSingle(inputElements, weightMatrices.get(k), numOutputElements);
			} else {
				outputList = processSingle(inputElements, weightMatrices.get(k), numInputElements);
			}
			
			inputElements = outputList;
		}
		
		return outputList;
	}

	public void train(TrainingSet trainingSet) {
		initWeights();
		
		for(int j = 0; j < numOutputElements; j++) {
			Double desiredOutput = 0.0;
			Double actualOutput = 0.0;
			
			long iter = 0;
			
			for(TrainingSample trainingSample : trainingSet.getTrainingSetSamples()) {
				
				desiredOutput = trainingSample.getOutputVector().get(j);
				actualOutput = process(trainingSample.getInputVector()).get(j);
				
				iter = 0;
				
				while(Math.pow(desiredOutput - actualOutput, 2) > minError && iter < maxIterations) {
					
					for(int k = 0; k < numLayers; k++) {
						
						RealMatrix weightMatrix = weightMatrices.get(k);
						
						double [] weights = weightMatrix.getColumn(j);
						
						for(int i = 0; i < weights.length; i++) {
							
							weightMatrix.setEntry(i, j, weights[i] + learningRate * (desiredOutput - actualOutput) * trainingSample.getInputVector().get(i));
							
						}
					}
					
					desiredOutput = trainingSample.getOutputVector().get(j);
					actualOutput = process(trainingSample.getInputVector()).get(j);
					iter++;
				}
				
			}
		}
	}

	public int getNumInputElements() {
		return numInputElements;
	}

	public void setNumInputElements(int numInputElements) {
		this.numInputElements = numInputElements;
	}

	public int getNumOutputElements() {
		return numOutputElements;
	}

	public void setNumOutputElements(int numOutputElements) {
		this.numOutputElements = numOutputElements;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public double getMinError() {
		return minError;
	}

	public void setMinError(double minError) {
		this.minError = minError;
	}

	public double getBias() {
		return bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public long getMaxIterations() {
		return maxIterations;
	}

	public void setMaxIterations(long maxIterations) {
		this.maxIterations = maxIterations;
	}

	public List<RealMatrix> getWeightMatrices() {
		return weightMatrices;
	}

	public void setWeightMatrices(List<RealMatrix> weightMatrices) {
		this.weightMatrices = weightMatrices;
	}

	public List<Double> getOutputElements() {
		return outputElements;
	}

	public void setOutputElements(List<Double> outputElements) {
		this.outputElements = outputElements;
	}
	
	public int getNumLayers() {
		return this.numLayers;
	}
	
	public void setNumLayers(int numLayers) {
		this.numLayers = numLayers;
	}
	
	public RealMatrix getWeightMatrix() {
		return this.weightMatrices.get(0);
	}
	
	public void setWeightMatrix(RealMatrix weightMatrix) {
		// NOOP
	}
	
}
