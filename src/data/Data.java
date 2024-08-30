package data;

import java.util.HashMap;


public class Data {
	private final int numberOfDimensions;
	private final int numberOfPoints;
	private final HashMap<Integer,String> dimensionNumberAndItsName;
	private final DataPoint[] points;
	private final DataStatistics dataStats;
	
	public Data(DataPoint[] points, int numberOfPoints, int numberOfDimensions, DataStatistics dataStats, HashMap<Integer,String> dimensionNumberAndItsName)
	{
//		System.out.println(numberOfDimensions);
		DataPoint.setNumberOfDimensions(numberOfDimensions);
		
//		System.out.println("Normalising data...");
		this.points = DataNormalisation.normalise(points, dataStats);
//		System.out.println("Normalisation - done.");
		
		this.numberOfPoints = numberOfPoints;
		this.numberOfDimensions = numberOfDimensions;
		this.dataStats = dataStats;
		this.dimensionNumberAndItsName = dimensionNumberAndItsName;
	}
	
	public HashMap<Integer, String> getDimensionNumberAndItsName() {
		return dimensionNumberAndItsName;
	}

	public int getNumberOfDimensions() {
		return numberOfDimensions;
	}
	public int getNumberOfPoints() {
		return numberOfPoints;
	}
	public DataPoint[] getPoints() {
		return points;
	}

	public DataStatistics getDataStats() {
		return dataStats;
	}
}
