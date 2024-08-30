package distance.measures;

import java.util.LinkedList;

import utils.Utils;
import Jama.Matrix;
import data.Cluster;
import data.DataPoint;
import data.Parameters;

public class GMMBayesMLE implements Measure {
	private static final double revOfSqrtOfPoweredDoublePi = (double)(1.0d/Math.sqrt(Math.pow(Math.PI*2, (double)DataPoint.getNumberOfDimensions())));// TODO moze tyu nie potrzeba tego dzielenia przez dwa (2.0d)? (double)(1.0d/Math.sqrt(Math.pow(Math.PI*2, (double)DataPoint.getNumberOfDimensions()/2.0d)));
	
	@Override
	public double distance(Cluster cluster, DataPoint point) { //funkcja gestosci rozkladu normalnego (pdf)
		double covarianceDet = cluster.getCovariance().det();
		if(Double.isInfinite(covarianceDet) || Double.isNaN(covarianceDet))
		{
			System.out.println("GMMBayesMLE.distance covarianceDet is not finite! covarianceDet = " + covarianceDet);
		}
		
		Matrix covarianceInversion;
		if(Utils.isDoubleExtremelyNearZero(covarianceDet))
		{
			cluster = makeCovMatrixInvertible(cluster);
			covarianceDet = cluster.getCovariance().det();
			//System.out.println("Made cov. matrix invertible!" + cluster);
			if(Double.isInfinite(covarianceDet) || Double.isNaN(covarianceDet))
			{
				System.out.println("GMMBayesMLE.distance covarianceDet is not finite after invocation of makeCovMatrixInvertible! covarianceDet = " + covarianceDet);
			}
		}
		else if(covarianceDet < 0)
		{
			System.err.println("GMMBayesMLE.distance Covariance matrix determinant is LOWER than 0!");
			System.err.println("Covariance matrix in cluster:");
			System.err.println("\n" + cluster.getCovariance().get(0, 0) + " " + cluster.getCovariance().get(0, 1));
			System.err.println(cluster.getCovariance().get(1, 0) + " " + cluster.getCovariance().get(1, 1));
			System.err.println("Covariance Det: " + covarianceDet);
			System.exit(1);
		}
		covarianceInversion = cluster.getCovariance().inverse();
		
		
		double firstFactor = revOfSqrtOfPoweredDoublePi / Math.sqrt(covarianceDet);
		
		Matrix x = point.getMatrix();
		Matrix miu = cluster.getCenter().getMatrix();
		Matrix expFunctionExponent = x.minus(miu); //TODO dla lepszego debuggingu rozpisuje krokami
		x = null;
		miu = null;
		
		Matrix notTransposedFactor = expFunctionExponent.copy();
		expFunctionExponent = expFunctionExponent.transpose();
		expFunctionExponent = expFunctionExponent.times(covarianceInversion);
		covarianceInversion = null;
		expFunctionExponent = expFunctionExponent.times(notTransposedFactor);
		notTransposedFactor = null;
		expFunctionExponent = expFunctionExponent.times(-0.5);
		
		if((expFunctionExponent.getColumnDimension() != expFunctionExponent.getRowDimension()) && expFunctionExponent.getRowDimension() != 1)
		{
			System.out.println("GMMBayesMLE.distance: after matrix operations returned value is not a scalar! column: "
					+ expFunctionExponent.getColumnDimension() + " row: " + expFunctionExponent.getRowDimension());
		}
		
		double expFunctionExponentValue = expFunctionExponent.get(0, 0);
		expFunctionExponent = null;
		double secondFactor = Math.exp(expFunctionExponentValue);
		double returnValue = (firstFactor*secondFactor);
		if(Double.isInfinite(returnValue) || Double.isNaN(returnValue))
		{
			System.out.println("GMMBayesMLE.distance calculated distance is not finite! Return value: " + returnValue
					+ " first factor: " + firstFactor + " second: " + secondFactor);
		}
		if(Parameters.isVerbose() && returnValue == 0.0)
		{
			System.out.println("GMMBayesMLE.distance calculated distance equals exactly 0.0! Return value: " + returnValue
					+ " first factor: " + firstFactor + " second: " + secondFactor + ". It is not good, consider "
					+ "increasing eplsilon and/ord little value. (expFunctionExponent.get(0, 0) = " + expFunctionExponentValue);
		}
		return returnValue;
	}

	@Override
	public double distance(DataPoint d1, DataPoint d2) {
		return 0;
	}

	protected Cluster makeCovMatrixInvertible(Cluster cluster) {//TODO mo�na rozkmini� SVD i doac wartosc tylko do tego elementu na diagonalu, gdzie nalezy
		Matrix covariance = cluster.getCovariance();
		double sumOfAddedValues = 0.0d;
		int increaseCounter = 0;
		while(Utils.isDoubleExtremelyNearZero(covariance.det()))
		{
			sumOfAddedValues += Parameters.getLittleValue();
			increaseCounter++;
			for(int i = 0; i < DataPoint.getNumberOfDimensions(); i++)
			{
				double covMatrixDiagValue = covariance.get(i, i);
				covMatrixDiagValue += Parameters.getLittleValue();
				covariance.set(i, i, covMatrixDiagValue);
			}
		}
		if(Parameters.isVerbose())
		{
			System.err.println(" Cov. matrix diagonal was increased by " + sumOfAddedValues
					+ " number of increasings: " + increaseCounter);
		}
		return cluster;
	}

	@Override
	public double[] updateCenter(DataPoint[] points) {
		// Not needed
		return null;
	}

	@Override
	public double calculateClusterisationStatistic(Cluster[] clusters) {
		LinkedList<DataPoint> allPoints = new LinkedList<DataPoint>();
		for(Cluster cls: clusters)
		{
			for(int i = 0; i < cls.getPoints().length; i++)
			{
				allPoints.add(cls.getPoints()[i]);
			}
		}
		
		double sumOfLog = 0.0d;
		double sumOfProb = 0.0d;
		for(DataPoint p: allPoints)
		{
			sumOfProb = 0.0d;
			for(Cluster cls: clusters)
			{
				sumOfProb += (distance(cls, p)*cls.getMixingCoefficient());
			}
			sumOfLog += Math.log(sumOfProb);
		}
		return sumOfLog;
	}

	@Override
	public String printClusterisationStatisticName() {
		return "Data Log. Likelihood";
	}

	@Override
	public int compareStatistics(double clusterisationStatistic,
			double clusterisationStatistic2) {
		if(clusterisationStatistic > clusterisationStatistic2)
		{
			return Measure.AFTER;
		}
		else if(clusterisationStatistic < clusterisationStatistic2)
		{
			return Measure.BEFORE;
		}
		else
		{
			return Measure.EQUAL;
		}
	}

	@Override
	public double getTheWorstMeasureValue() {
		return (-1)*Double.MAX_VALUE;
	}
}
