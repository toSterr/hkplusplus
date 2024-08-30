package center.method;

import data.Cluster;
import data.Data;
import data.DataPoint;
import distance.measures.Measure;

public interface CenterMethod {

	Cluster makeCluster(Data points, Measure measure);

    Cluster makeCluster(DataPoint[] points, Measure measure);

    Cluster updateCenter(Cluster cluster, Measure measure);

}
