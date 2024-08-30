package distance.measures;

import data.Cluster;
import data.DataPoint;

public interface Measure {
	public final int BEFORE = -1;
	public final int EQUAL = 0;
	public final int AFTER = 1;
    
	public double getTheWorstMeasureValue();
	
	public double distance(Cluster center, DataPoint d2);

	public double distance(DataPoint d1, DataPoint d2);

	public double[] updateCenter(DataPoint[] points);

	public double calculateClusterisationStatistic(Cluster[] clusters);

	public String printClusterisationStatisticName();

	public int compareStatistics(double clusterisationStatistic,
			double clusterisationStatistic2);
	
}
