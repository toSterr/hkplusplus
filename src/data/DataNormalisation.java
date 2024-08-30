package data;

public class DataNormalisation {
	public static DataPoint[] normalise(DataPoint[] inputData, DataStatistics dataStats)
	{
		examineData(inputData);
		
		DataPoint[] normalisedData = null;
		normalisedData = transformData(inputData, dataStats);
		
		return normalisedData;
	}

	private static DataPoint[] transformData(DataPoint[] inputData,
			DataStatistics dataStats) {
		DataPoint[] transformedData = new DataPoint[inputData.length];
		
		for(int i = 0; i < transformedData.length; i++)
		{
			double[] normalisedCoordinates = new double[DataPoint.getNumberOfDimensions()];
			for(int j = 0; j < DataPoint.getNumberOfDimensions(); j++)
			{
				if(dataStats.getMinValues()[j] != dataStats.getMaxValues()[j])
				{
					normalisedCoordinates[j] = (inputData[i].getCoordinate(j) - dataStats.getMinValues()[j])
							/ (dataStats.getMaxValues()[j] - dataStats.getMinValues()[j]);
				}
				else if(dataStats.getMaxValues()[j] == dataStats.getMinValues()[j])
				{
					normalisedCoordinates[j] = 0;
				}
				else
				{
					System.err.println("Something went wrong with dataStats.getMaxValues()[j] and/or getMinValues()[j]"
							+ " Values are: " + dataStats.getMaxValues()[j] + " and "+ dataStats.getMinValues()[j]);
					System.exit(1);
				}
			}
			transformedData[i] = new DataPoint(normalisedCoordinates, inputData[i].getSourceCoordinates(), inputData[i].getInstanceName(),
					inputData[i].getClassAttribute());
			transformedData[i].setGlobalUsageCounter(inputData[i].getGlobalUsageCounter());
		}
		return transformedData;
	}

	private static void examineData(DataPoint[] inputData) {
		if(inputData == null)
		{
			System.err.println("DataNormalisation.normalise(..) Can't normalise data, inputData is null.");
			System.exit(1);
		}
		else if(inputData.length == 0)
		{
			System.err.println("DataNormalisation.normalise(..) Can't normalise data, inputData has 0 Data Points.");
			System.exit(1);
		}
		else if(inputData[0] == null)
		{
			for(int i = 0; i < inputData.length; i++)
			{
				if(inputData[i] == null)
				{
					System.err.println("DataNormalisation.normalise(..) Can't normalise data, inputData[" + i + "] is null.");
					System.exit(1);
				}
			}
		}
		else if(DataPoint.getNumberOfDimensions() <= 0)
		{
			System.err.println("DataNormalisation.normalise(..) Can't normalise data, DataPoint.getNumberOfDimensions() equals "
					+ DataPoint.getNumberOfDimensions() + ".");
			System.exit(1);
		}
	}
}
