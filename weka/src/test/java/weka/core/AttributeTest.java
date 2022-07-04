package weka.core;

import lombok.extern.slf4j.Slf4j;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * Weka中五种不同的属性类型.
 *
 * @author caogaoli
 * @date 2022/7/4 16:17
 */
@Slf4j
public class AttributeTest {

    @Test
    public void tesNumeric() {
        // 数值型
        Attribute numeric = new Attribute("attribute_name");
        Assert.assertTrue(numeric.isNumeric());
    }

    @Test
    public void testDate() {
        // 时间型
        Attribute date = new Attribute("attribute_name", "yyyy-MM-dd");
        Assert.assertTrue(date.isDate());
    }

    @Test
    public void testLabels() {
        // 标签型
        List<String> labels = new ArrayList<>();
        labels.add("label_a");
        labels.add("label_b");
        labels.add("label_c");
        labels.add("label_d");
        Attribute nominal = new Attribute("attribute_name", labels);
        Assert.assertTrue(nominal.isNominal());
    }

    @Test
    public void testString() {
        // 字符串型
        // 与标称（标签）属性不同，字符串类型不需要存放预定义的标签列表，通常用于存储额文本数据，即文本分类的文档内容。
        // 字符串型使用与标称型属性相同的构造函数，但需要提供一个null值，而非java.util.ArrayList<String>的实例.
        Attribute string = new Attribute("attribute_name", (ArrayList<String>) null);
        Assert.assertTrue(string.isString());
    }

    @Test
    public void testRelational() {
        // 关系型.
        ArrayList<Attribute> attrs = new ArrayList<Attribute>();
        attrs.add(new Attribute("rel.num"));
        List<String> values = new ArrayList<>();
        values.add("val_A");
        values.add("val_B");
        values.add("val_C");
        attrs.add(new Attribute("rel.nom", values));
        Instances rel_structure = new Instances("rel", attrs, 0);
        Attribute relational = new Attribute("attribute_name", rel_structure);
        Assert.assertTrue(relational.isRelationValued());
    }

}
