package distance.measures;

import java.util.LinkedList;

import Jama.Matrix;
import data.Cluster;
import data.DataPoint;
import data.Parameters;
import utils.Utils;

public class LOG_GMMBayesMLE extends GMMBayesMLE implements Measure {
	private final static double minusHalfOfDimMultipliedByLogOfTWO = (-1.0)*(DataPoint.getNumberOfDimensions()/2.0)*Math.log(2);
	private final static double minusHalfOfDimMultipliedByLogOfPI = (-1.0)*(DataPoint.getNumberOfDimensions()/2.0)*Math.log(Math.PI);
	
	@Override
	public double distance(Cluster cluster, DataPoint point) { //funkcja gestosci rozkladu normalnego (pdf)
		double covarianceDet = cluster.getCovariance().det(); // TODO: we can save det in the cluster object instead of computing it all the time - it is n^3!
		if(Double.isInfinite(covarianceDet) || Double.isNaN(covarianceDet))
		{
			System.out.println("LOG_GMMBayesMLE.distance covarianceDet is not finite! covarianceDet = " + covarianceDet);
		}
		
		Matrix covarianceInversion;
		if(Utils.isDoubleExtremelyNearZero(covarianceDet))
		{
			cluster = makeCovMatrixInvertible(cluster);
			covarianceDet = cluster.getCovariance().det();
			//System.out.println("Made cov. matrix invertible!" + cluster);
			if(Double.isInfinite(covarianceDet) || Double.isNaN(covarianceDet))
			{
				System.out.println("LOG_GMMBayesMLE.distance covarianceDet is not finite after invocation of makeCovMatrixInvertible! covarianceDet = " + covarianceDet);
			}
		}
		else if(covarianceDet < 0)
		{
			System.err.println("LOG_GMMBayesMLE.distance Covariance matrix determinant is LOWER than 0!");
			System.err.println("Covariance matrix in cluster:");
			System.err.println("\n" + cluster.getCovariance().get(0, 0) + " " + cluster.getCovariance().get(0, 1));
			System.err.println(cluster.getCovariance().get(1, 0) + " " + cluster.getCovariance().get(1, 1));
			System.err.println("Covariance Det: " + covarianceDet);
			System.exit(1);
		}
		covarianceInversion = cluster.getCovariance().inverse(); // TODO: we can save cov matrix inverse in the cluster object instead of computing it all the time - it is n^3!
		
		
		double firstPartValue = minusHalfOfDimMultipliedByLogOfTWO + minusHalfOfDimMultipliedByLogOfPI - 0.5*Math.log(covarianceDet);
		
		Matrix x = point.getMatrix();
		Matrix miu = cluster.getCenter().getMatrix();
		Matrix secondPart = x.minus(miu); //TODO dla lepszego debuggingu rozpisuje krokami
		x = null;
		miu = null;
		
		Matrix notTransposedFactor = secondPart.copy();
		secondPart = secondPart.transpose();
		secondPart = secondPart.times(covarianceInversion);
		covarianceInversion = null;
		secondPart = secondPart.times(notTransposedFactor);
		notTransposedFactor = null;
		secondPart = secondPart.times(-0.5);
		
		if((secondPart.getColumnDimension() != secondPart.getRowDimension()) && secondPart.getRowDimension() != 1)
		{
			System.out.println("LOG_GMMBayesMLE.distance: after matrix operations returned value is not a scalar! column: "
					+ secondPart.getColumnDimension() + " row: " + secondPart.getRowDimension());
		}
		
		double secondPartValue = secondPart.get(0, 0);
		secondPart = null;
		double returnValue = (firstPartValue + secondPartValue);
		if(Double.isInfinite(returnValue) || Double.isNaN(returnValue))
		{
			System.out.println("LOG_GMMBayesMLE.distance calculated distance is not finite! Return value: " + returnValue
					+ " firstPart: " + firstPartValue + " secondPartValue: " + secondPartValue);
		}
		if(Parameters.isVerbose() && returnValue == 0.0)
		{
			System.out.println("LOG_GMMBayesMLE.distance calculated distance equals exactly 0.0! Return value: " + returnValue
					+ " firstPartValuer: " + firstPartValue + " secondPartValue: " + secondPartValue + ". It is not good, consider "
					+ "increasing eplsilon and/ord little value.");
		}
		return returnValue;
	}

	@Override
	public double calculateClusterisationStatistic(Cluster[] clusters) {
		LinkedList<DataPoint> allPoints = new LinkedList<DataPoint>();
		double[] logOfMixingCoefficients = new double[clusters.length];
		for(int i = 0; i < clusters.length; i++)
		{
			logOfMixingCoefficients[i] = Math.log(clusters[i].getMixingCoefficient());
			for(int j = 0; j < clusters[i].getPoints().length; j++)
			{
				allPoints.add(clusters[i].getPoints()[j]);
			}
		}
		
		double[][] pointsToMixturesLogOfProbabilitiy = new double[allPoints.size()][clusters.length];
		for(int i = 0; i < allPoints.size(); i++)
		{
			for(int j = 0; j < clusters.length; j++)
			{
				pointsToMixturesLogOfProbabilitiy[i][j] = distance(clusters[j], allPoints.get(i));
			}
		}
		
		double sumOfLogSumExpTrick = 0.0d;
		double sumOfProb = 0.0d;
		double max = (-1)*Double.MAX_VALUE;
		double[] particularPointToMisturesLogOfResponsibility;
		for(int i = 0; i < allPoints.size(); i++)
		{
			particularPointToMisturesLogOfResponsibility = new double[clusters.length];
			max = (-1)*Double.MAX_VALUE;
			for(int j = 0; j < clusters.length; j++)
			{
				particularPointToMisturesLogOfResponsibility[j] = logOfMixingCoefficients[j] + pointsToMixturesLogOfProbabilitiy[i][j];
				if(particularPointToMisturesLogOfResponsibility[j] >= max)
				{
					max = particularPointToMisturesLogOfResponsibility[j];
				}
			}
			
			sumOfProb = 0.0d;
			for(int j = 0; j < clusters.length; j++)
			{
				double exponent = particularPointToMisturesLogOfResponsibility[j] - max;
				sumOfProb += Math.exp(exponent);
			}
			
			sumOfLogSumExpTrick += max + Math.log(sumOfProb);
		}
		if(Double.isInfinite(sumOfLogSumExpTrick))
		{
			if(Parameters.isVerbose())
			{
				System.err.println("LOG_GMMBayesMLE.calculateClusterisationStatistic() pobabilities of points to clusters is that small that sum of it is zero!"
						+ " And the Log(0) is not defined. If it is happening at the bottom of the hierarchy then it is ok (because of little objects to estimate"
						+ " covariance), but OTHERWISE THERE IS A BUG!\nReturning maximum negative value.");
			}
			sumOfLogSumExpTrick = (-1)*(Double.MAX_VALUE);
		}
		return sumOfLogSumExpTrick;
	}
}
