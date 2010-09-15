import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

public class Adaline implements Network {
	private int numInputElements;
	private int numOutputElements;

	private double learningRate;
	private double minError;
	private double bias;
	private long maxIterations;

	private RealMatrix weightMatrix;
	
	private List<Double> outputElements = new ArrayList<Double>();

	public Adaline() {
		initWeights();
	}
	
	/* (non-Javadoc)
	 * @see Network#getBias()
	 */
	public double getBias() {
		return bias;
	}

	/* (non-Javadoc)
	 * @see Network#setBias(double)
	 */
	public void setBias(double bias) {
		this.bias = bias;
	}

	protected void initWeights() {
		
		double [][] d = new double[numInputElements][numOutputElements];
		
		for(int i = 0; i < numInputElements; i++) {
			for(int j = 0; j < numOutputElements; j++) {
				d[i][j] = Math.random();
			}
		}
		
		weightMatrix = new Array2DRowRealMatrix(d);
		
	}
	
	/* (non-Javadoc)
	 * @see Network#getLearningRate()
	 */
	public double getLearningRate() {
		return learningRate;
	}

	/* (non-Javadoc)
	 * @see Network#setLearningRate(double)
	 */
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	/* (non-Javadoc)
	 * @see Network#getMinError()
	 */
	public double getMinError() {
		return minError;
	}

	/* (non-Javadoc)
	 * @see Network#setMinError(double)
	 */
	public void setMinError(double minError) {
		this.minError = minError;
	}

	/* (non-Javadoc)
	 * @see Network#getMaxIterations()
	 */
	public long getMaxIterations() {
		return maxIterations;
	}

	/* (non-Javadoc)
	 * @see Network#setMaxIterations(long)
	 */
	public void setMaxIterations(long maxIterations) {
		this.maxIterations = maxIterations;
	}

	/* (non-Javadoc)
	 * @see Network#getNumInputElements()
	 */
	public int getNumInputElements() {
		return numInputElements;
	}

	/* (non-Javadoc)
	 * @see Network#setNumInputElements(int)
	 */
	public void setNumInputElements(int numInputElements) {
		this.numInputElements = numInputElements;
	}

	/* (non-Javadoc)
	 * @see Network#getNumOutputElements()
	 */
	public int getNumOutputElements() {
		return numOutputElements;
	}

	/* (non-Javadoc)
	 * @see Network#setNumOutputElements(int)
	 */
	public void setNumOutputElements(int numOutputElements) {
		this.numOutputElements = numOutputElements;
	}

	/* (non-Javadoc)
	 * @see Network#getWeightMatrix()
	 */
	public RealMatrix getWeightMatrix() {
		return weightMatrix;
	}

	/* (non-Javadoc)
	 * @see Network#setWeightMatrix(org.apache.commons.math.linear.RealMatrix)
	 */
	public void setWeightMatrix(RealMatrix weightMatrix) {
		this.weightMatrix = weightMatrix;
	}

	/* (non-Javadoc)
	 * @see Network#getOutputElements()
	 */
	public List<Double> getOutputElements() {
		return outputElements;
	}

	/* (non-Javadoc)
	 * @see Network#setOutputElements(java.util.List)
	 */
	public void setOutputElements(List<Double> outputElements) {
		this.outputElements = outputElements;
	}

	/* (non-Javadoc)
	 * @see Network#process(java.util.List)
	 */
	public List<Double> process(List<Double> inputElements) {
		
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
	
	private Double sigmoid(Double value) {
		return 1.0 / (1 + Math.exp(-1000 * value));
	}
	
	/* (non-Javadoc)
	 * @see Network#train(TrainingSet)
	 */
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
					
					double [] weights = weightMatrix.getColumn(j);
					
					for(int i = 0; i < weights.length; i++) {
						
						weightMatrix.setEntry(i, j, weights[i] + learningRate * (desiredOutput - actualOutput) * trainingSample.getInputVector().get(i));
						
					}
					
					desiredOutput = trainingSample.getOutputVector().get(j);
					actualOutput = process(trainingSample.getInputVector()).get(j);
					iter++;
				}
				
			}
		}
		
	}
}
