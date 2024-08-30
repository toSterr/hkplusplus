package distance.measures;

import data.Cluster;
import data.DataPoint;

public class L2Norm implements Measure {

	@Override
	public double distance(Cluster d1, DataPoint d2) {
		double distance = 0.0d;
		double difference;
		for(int i = 0; i < d1.getCenter().getCoordinates().length; i++)
		{
			difference = d1.getCenter().getCoordinate(i) - d2.getCoordinate(i);
			distance += difference*difference;
		}
		return Math.sqrt(distance);
	}

	@Override
	public double distance(DataPoint d1, DataPoint d2) {
		double distance = 0.0d;
		double difference;
		for(int i = 0; i < d1.getCoordinates().length; i++)
		{
			difference = d1.getCoordinate(i) - d2.getCoordinate(i);
			distance += difference*difference;
		}
		return Math.sqrt(distance);
	}

	@Override
	public double[] updateCenter(DataPoint[] points) {
		double[] centroid = sumEachDimensionPointsCoordinates(points);
		centroid = averageSums(points, centroid);
		
		return centroid;
	}

	private double[] sumEachDimensionPointsCoordinates(DataPoint[] points) {
		double[] centroid = new double[DataPoint.getNumberOfDimensions()];
		for(DataPoint p: points)
		{
			for(int i = 0; i < DataPoint.getNumberOfDimensions(); i++)
			{
				centroid[i] += p.getCoordinate(i);
			}
		}
		return centroid;
	}

	private double[] averageSums(DataPoint[] points, double[] centroid) {
		for(int i = 0; i < DataPoint.getNumberOfDimensions(); i++)
		{
			centroid[i] /= (double) points.length;
		}
		return centroid;
	}

	@Override
	public double calculateClusterisationStatistic(Cluster[] clusters) {
		double sumOfInterClustersDistance = 0.0d;
		for(Cluster cls: clusters)
		{
			sumOfInterClustersDistance += particularClusterInterDistance(cls);
		}
		return sumOfInterClustersDistance;
	}
	
	public double particularClusterInterDistance(Cluster cluster) {
		double interClusterDistance = 0.0d;
		for(DataPoint point: cluster.getPoints())
		{
			interClusterDistance += distance(cluster, point);
		}
		return interClusterDistance;
	}

	@Override
	public int compareStatistics(double clusterisationStatistic,
			double clusterisationStatistic2) {
		if(clusterisationStatistic < clusterisationStatistic2)
		{
			return Measure.AFTER; 
		}
		else if(clusterisationStatistic > clusterisationStatistic2)
		{
			return Measure.BEFORE;
		}
		else
		{
			return Measure.EQUAL;
		}
	}
	
	@Override
	public String printClusterisationStatisticName() {
		return "Sum of Inter cluster distances";
	}

	@Override
	public double getTheWorstMeasureValue() {
		return Double.MAX_VALUE;
	}

}
