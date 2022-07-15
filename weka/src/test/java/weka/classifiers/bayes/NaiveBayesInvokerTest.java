package weka.classifiers.bayes;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

@Slf4j
public class NaiveBayesInvokerTest {

    @Test
    public void testWeatherClusterByBayesNet() throws Exception {
        // 加载数据
        String file = "data/weather.nominal.arff";
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(file));
        Instances dataSet = loader.getDataSet();
        dataSet.setClassIndex(dataSet.numAttributes() - 1);
        log.info("data set cluster attribute index: {}", dataSet.numAttributes() - 1);
        // 分类器
        NaiveBayes naiveBayes = new NaiveBayes();
        // 一次性加载训练
        naiveBayes.buildClassifier(dataSet);
        log.info("naive bays {}", naiveBayes);
    }

}
