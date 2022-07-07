package weka.attributeSelection;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.util.Arrays;
import java.util.Random;

/**
 * attribute selection operation.
 *
 * @author caogaoli
 * @date 2022/7/7 13:29
 */
@Slf4j
public class AttributeSelectionInvokerTest {

    // 使用元分类器进行属性选择.
    @Test
    public void testUseMetaClassifier() throws Exception {
        // 加载数据
        String file = "data/weather.numeric.arff";
        Instances instances = ConverterUtils.DataSource.read(file);
        // 设置类别的属性
        if (instances.classIndex() == -1) {
            instances.setClassIndex(instances.numAttributes() - 1);
        }
        // 评估器
        CfsSubsetEval evaluation = new CfsSubsetEval();
        // 搜索算法，并设置成向后模式
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        // 分类器
        J48 baseClassifier = new J48();
        // 元分类器，并给元分类器进行配置重要组件.
        // 该对象的功能是，在将数据传递给分类器之前，用属性选择降低训练数据和测试数据的维度.
        AttributeSelectedClassifier metaClassifier = new AttributeSelectedClassifier();
        metaClassifier.setClassifier(baseClassifier);
        metaClassifier.setEvaluator(evaluation);
        metaClassifier.setSearch(search);
        Evaluation evaluationModel = new Evaluation(instances);
        int folds = 10;
        // 交叉验证模型.
        evaluationModel.crossValidateModel(metaClassifier, instances, folds, new Random(1234L));
        log.info("model {}", evaluationModel.toSummaryString());
    }

    // 使用过滤器进行属性的选择.
    @Test
    public void testUseFilter() throws Exception {
        // 加载数据.
        String file = "data/weather.numeric.arff";
        Instances instances = ConverterUtils.DataSource.read(file);
        log.info("origin attribute size {}", instances.numAttributes());
        log.info("origin instances size {}", instances.numInstances());
        // 设置类别属性的索引
        if (instances.numInstances() == -1) {
            instances.setClassIndex(instances.numAttributes() - 1);
        }
        // 开始装配的属性过滤器.
        AttributeSelection filter = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(instances);
        Instances result = Filter.useFilter(instances, filter);
        log.info("finally attribute size {}", result.numAttributes());
        log.info("finally instances size {}", result.numInstances());
        log.info("result {}", result);
    }

    @Test
    public void testUseLowLevel() throws Exception {
        String file = "data/weather.numeric.arff";
        Instances instances = ConverterUtils.DataSource.read(file);
        if (instances.classIndex() == -1) {
            instances.setClassIndex(instances.numAttributes() - 1);
        }
        weka.attributeSelection.AttributeSelection selection = new weka.attributeSelection.AttributeSelection();
        // 设置搜索器
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        selection.setSearch(search);
        // 设置评价器
        CfsSubsetEval eval = new CfsSubsetEval();
        selection.setEvaluator(eval);
        // 在instances上执行属性select操作.
        selection.SelectAttributes(instances);
        // 获取属性选择后的结果.
        int[] items = selection.selectedAttributes();
        log.info("选择属性索引(从0开始)：\n {}", Arrays.toString(items));
    }

}
