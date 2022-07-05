package weka.classifiers.trees;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.util.Random;

/**
 * J48分类器训练.
 *
 * @author caogaoli
 * @date 2022/7/5 10:51
 */
@Slf4j
public class J48WeatherDataTest {

    private static final String FILE_PATH = "src/test/resources/data/weather.nominal.arff";

    // 批量训练
    @Test
    public void testWeatherArff() throws Exception {
        // 加载数据
        ArffLoader loader = new ArffLoader();
        File file = new File(FILE_PATH);
        loader.setFile(file);
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        // 训练J48分类器
        String[] options = new String[1];
        // 未裁剪树选项
        options[0] = "-U";
        // J48分类器对象
        J48 tree = new J48();
        // 设置选项卡
        tree.setOptions(options);
        // 构建分类器
        tree.buildClassifier(data);
        // 输出生成的模型
        log.info("未裁剪决策树 {}", tree);
    }

    // 增量方式构建NaiveBayes分类器，并输出生成的模型.
    @Test
    public void testWeatherArffIncrementClassifier() throws Exception {
        File file = new File(FILE_PATH);
        // 加载数据
        ArffLoader loader = new ArffLoader();
        loader.setFile(file);
        Instances structure = loader.getStructure();
        structure.setClassIndex(structure.numAttributes() - 1);
        //
        NaiveBayesUpdateable naiveBayesUpdateable = new NaiveBayesUpdateable();
        naiveBayesUpdateable.buildClassifier(structure);
        Instance instance;
        while ((instance = loader.getNextInstance(structure)) != null) {
            naiveBayesUpdateable.updateClassifier(instance);
        }
        log.info("{}", naiveBayesUpdateable);
    }

    @Test
    public void testWeatherArffWithEvaluation() throws Exception {
        // 加载数据
        ArffLoader loader = new ArffLoader();
        File file = new File(FILE_PATH);
        loader.setFile(file);
        Instances dataSet = loader.getDataSet();
        dataSet.setClassIndex(dataSet.numAttributes() - 1);
        Evaluation evaluation = new Evaluation(dataSet);
        J48 tree = new J48();
        Random random = new Random(1234L);
        evaluation.crossValidateModel(tree, dataSet, 10, random);
        log.info(evaluation.toSummaryString("result\n", true));
    }

}
