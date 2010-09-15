import java.io.File;
import java.io.FileReader;

import com.thoughtworks.xstream.XStream;


public class AdalineUtils
{
  public static Network readNetwork(File networkFile) throws Exception {
    
    XStream xstream = new XStream();
    
    Network network = (Network) xstream.fromXML( new FileReader( networkFile ) );
    
    return network;
  }
  
  public static TrainingSet readTrainingSet(File trainingSetFile) throws Exception {
    XStream xstream = new XStream();
    
    TrainingSet trainingSet = (TrainingSet) xstream.fromXML( new FileReader(trainingSetFile) );
    
    return trainingSet;
  }
}
