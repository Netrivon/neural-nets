import java.util.List;

import org.apache.commons.math.linear.RealMatrix;


public interface MultiLayerNetwork extends Network {
	public abstract List<RealMatrix> getWeightMatrices();
	public abstract void setWeightMatrices(List<RealMatrix> weightMatrices);
	public abstract void setNumLayers(int numLayers);
	public abstract int getNumLayers();
	
}