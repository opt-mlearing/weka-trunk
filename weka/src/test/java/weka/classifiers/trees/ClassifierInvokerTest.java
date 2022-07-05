package weka.classifiers.trees;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.supervised.attribute.AddClassification;

import java.io.File;
import java.util.Arrays;
import java.util.Random;

/**
 * J48分类器训练.
 *
 * @author caogaoli
 * @date 2022/7/5 10:51
 */
@Slf4j
public class ClassifierInvokerTest {

    private static final String FILE_PATH = "data/weather.nominal.arff";

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

    // 构建批量分类器
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

    // 构建增量分类器
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


    // 输出类别分布
    @Test
    public void testSegmentChallenge() throws Exception {
        String trainFile = "data/segment-challenge.arff";
        String testFile = "data/segment-test.arff";
        // 加载训练数据
        Instances train = ConverterUtils.DataSource.read(trainFile);
        // 加载测试数据
        Instances test = ConverterUtils.DataSource.read(testFile);
        // 设置类别索引
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);
        // 检查训练集和测试集是否兼容
        if (!train.equalHeaders(test)) {
            throw new Exception("训练集和测试集不兼容：" + train.equalHeaders(test));
        }
        // 训练分类器
        J48 tree = new J48();
        tree.buildClassifier(train);
        // 输出预测
        log.info("编号\t - 实际\t - 预测\t - 错误\t - 分布\t");
        // test数据集中的实例逐个预测.
        for (int i = 0; i < test.numInstances(); ++i) {
            Instance instance = test.get(i);
            // 获取预测值
            double predict = tree.classifyInstance(instance);
            // 获取预测分布
            double[] distribution = tree.distributionForInstance(instance);
            log.info("{} - {} - {} - {} - {} ", i + 1,
                    test.instance(i).toString(test.classIndex()),
                    test.classAttribute().value((int) predict),
                    predict == test.instance(i).classValue() ? "Y" : "N",
                    Arrays.toString(distribution));
        }
    }

    // 单次运行交叉验证
    @Test
    public void testRunOneCV() throws Exception {
        String filePath = "data/ionosphere.arff";
        // 读取数据
        Instances origin = ConverterUtils.DataSource.read(filePath);
        // 获取类别索引
        origin.setClassIndex(origin.numAttributes() - 1);
        // 选项卡配置
        String[] options = new String[2];
        // 默认参数
        options[0] = "-C";
        options[1] = "0.25";
        String className = "weka.classifiers.trees.J48";
        // 通过反射获取分类器实例.
        Classifier classifier = (Classifier) Utils.forName(Classifier.class, className, options);
        // 验证实验参数设置
        Random random = new Random(1234L);
        Instances data = new Instances(origin);
        // 设置随机因子
        data.randomize(random);
        // 执行交叉验证
        int folds = 10;
        Evaluation evaluation = new Evaluation(data);
        for (int i = 0; i < folds; ++i) {
            // 测试集
            Instances train = data.trainCV(folds, i);
            // 训练集
            Instances test = data.testCV(folds, i);
            Classifier tmp = AbstractClassifier.makeCopy(classifier);
            tmp.buildClassifier(train);
            evaluation.evaluateModel(tmp, test);
        }
        // 输出评估
        log.info("=== 分类器设置 ===");
        log.info("分类器 {}", Utils.toCommandLine(classifier));
        log.info("数据集 {}", data.relationName());
        log.info("折数 {}", folds);
        log.info("随机种子 {}", 1234L);
        log.info(evaluation.toSummaryString("=== " + folds + "折交叉验证 ===", false));
    }

    // 单次交叉验证，并将预测结果保留结果.
    @Test
    public void testCVPrediction() throws Exception {
        // 加载数据
        String filePath = "data/ionosphere.arff";
        File file = new File(filePath);
        ArffLoader loader = new ArffLoader();
        loader.setFile(file);
        Instances originDataSet = loader.getDataSet();
        // 设置类索引
        originDataSet.setClassIndex(originDataSet.numAttributes() - 1);
        // 分类器
        String[] options = new String[2];
        options[0] = "-C";
        options[1] = "0.25";
        String className = "weka.classifiers.trees.J48";
        // 通过反射获取分类器实例.
        Classifier classifier = (Classifier) Utils.forName(Classifier.class, className, options);
        // 验证实验参数设置
        Random random = new Random(1234L);
        Instances dataSet = new Instances(originDataSet);
        // 设置随机因子
        dataSet.randomize(random);
        // 折数
        int folds = 10;
        // 如果类别为标称型，则根据其类别值进行分层.
        if (dataSet.classAttribute().isNominal()) {
            dataSet.stratify(folds);
        }
        // 执行交叉验证，并进行预测
        Evaluation evaluation = new Evaluation(dataSet);
        // 预测数据
        Instances predictedData = null;
        for (int i = 0; i < folds; ++i) {
            // 训练集
            Instances train = dataSet.trainCV(folds, i);
            // 测试集
            Instances test = dataSet.testCV(folds, i);
            // 构建 & 评估 分类器
            Classifier tmp = AbstractClassifier.makeCopy(classifier);
            tmp.buildClassifier(train);
            evaluation.evaluateModel(tmp, test);
            // 添加预测
            AddClassification filter = new AddClassification();
            // 设置分类器
            filter.setClassifier(classifier);
            filter.setOutputClassification(true);
            filter.setOutputDistribution(true);
            filter.setOutputErrorFlag(true);
            filter.setInputFormat(train);
            // 训练分类器
            filter.useFilter(train, filter);
            // 在测试集合上进行预测
            Instances instances = filter.useFilter(test, filter);
            if (predictedData == null) {
                // 避免预数据集为空.
                // 初始化数据集头部.
                predictedData = new Instances(instances, 0);
            }
            //
            for (int j = 0; j < instances.numInstances(); ++j) {
                predictedData.add(instances.instance(j));
            }
        }
        // 评估结果
        log.info("=== 分类器设置 ===\n");
        // 分类器及其选项卡配置.
        if (classifier instanceof OptionHandler) {
            log.info("分类器 {}, 选项设置 {}",
                    classifier.getClass().getSimpleName(),
                    Utils.joinOptions(((OptionHandler) classifier).getOptions()));
        } else {
            log.info("分类器 {}", classifier.getClass().getSimpleName());
        }
        log.info("数据集 {}", dataSet.relationName());
        log.info("折数 {}", folds);
        log.info("随机种子 {}", 1234L);
        log.info(evaluation.toSummaryString("=== " + folds + " 折交叉验证=== ", false));
        // 写入数据文件，若该文件存在，则先删除再保留预测结果.
        String writeTarget = "files/predictions.arff";
        File writeObj = new File(writeTarget);
        if (writeObj.exists()) {
            writeObj.delete();
        }
        ConverterUtils.DataSink.write(writeTarget, predictedData);
    }

    // 多次运行交叉验证.
    @Test
    public void testRunTenTimesCV() throws Exception {
        String file = "data/labor.arff";
        // 获取并加载数据
        Instances originDataSource = ConverterUtils.DataSource.read(file);
        // 设置类别索引
        originDataSource.setClassIndex(originDataSource.numAttributes() - 1);
        String className = "weka.classifiers.trees.J48";
        String[] options = new String[2];
        options[0] = "-C";
        options[1] = "0.25";
        Classifier classifier = (Classifier) Utils.forName(Classifier.class, className, options);
        // 运行次数
        int runs = 10;
        // 对折数
        int folds = 10;
        for (int i = 0; i < runs; ++i) {
            long seed = 1234L + i;
            Random random = new Random(seed);
            Instances instances = new Instances(originDataSource);
            instances.randomize(random);
            // 如果类别为标称类型，则根据其类别值进行分层.
            if (instances.classAttribute().isNominal()) {
                // 根据类别进行分层
                instances.stratify(folds);
            }
            Evaluation evaluation = new Evaluation(instances);
            for (int j = 0; j < folds; ++j) {
                // 训练集
                Instances train = instances.trainCV(folds, i);
                // 测试集
                Instances test = instances.testCV(folds, i);
                Classifier copyClassifier = AbstractClassifier.makeCopy(classifier);
                // 构建分类器
                copyClassifier.buildClassifier(train);
                // 评估分类器
                evaluation.evaluateModel(copyClassifier, test);
            }
            log.info("=== 运行第 " + (i - 1) + " 次的分类器设置 ===");
            log.info("分类器 {}", Utils.toCommandLine(classifier));
            log.info("数据集 {}", originDataSource.relationName());
            log.info("对折次数 {}", folds);
            log.info("重跑次数 {}", runs);
            log.info(evaluation.toSummaryString("=== " + folds + "  折交叉验证运行第" + (i + 1) + " 次===", false));
        }
    }

}
