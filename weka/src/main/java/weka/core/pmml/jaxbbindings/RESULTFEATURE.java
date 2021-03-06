//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.0-b52-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2013.12.20 at 12:48:21 PM GMT 
//


package weka.core.pmml.jaxbbindings;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for RESULT-FEATURE.
 *
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="RESULT-FEATURE">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="predictedValue"/>
 *     &lt;enumeration value="predictedDisplayValue"/>
 *     &lt;enumeration value="transformedValue"/>
 *     &lt;enumeration value="decision"/>
 *     &lt;enumeration value="probability"/>
 *     &lt;enumeration value="affinity"/>
 *     &lt;enumeration value="residual"/>
 *     &lt;enumeration value="standardError"/>
 *     &lt;enumeration value="clusterId"/>
 *     &lt;enumeration value="clusterAffinity"/>
 *     &lt;enumeration value="entityId"/>
 *     &lt;enumeration value="entityAffinity"/>
 *     &lt;enumeration value="warning"/>
 *     &lt;enumeration value="ruleValue"/>
 *     &lt;enumeration value="reasonCode"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 */
@XmlEnum
public enum RESULTFEATURE {

    @XmlEnumValue("affinity")
    AFFINITY("affinity"),
    @XmlEnumValue("clusterAffinity")
    CLUSTER_AFFINITY("clusterAffinity"),
    @XmlEnumValue("clusterId")
    CLUSTER_ID("clusterId"),
    @XmlEnumValue("decision")
    DECISION("decision"),
    @XmlEnumValue("entityAffinity")
    ENTITY_AFFINITY("entityAffinity"),
    @XmlEnumValue("entityId")
    ENTITY_ID("entityId"),
    @XmlEnumValue("predictedDisplayValue")
    PREDICTED_DISPLAY_VALUE("predictedDisplayValue"),
    @XmlEnumValue("predictedValue")
    PREDICTED_VALUE("predictedValue"),
    @XmlEnumValue("probability")
    PROBABILITY("probability"),
    @XmlEnumValue("reasonCode")
    REASON_CODE("reasonCode"),
    @XmlEnumValue("residual")
    RESIDUAL("residual"),
    @XmlEnumValue("ruleValue")
    RULE_VALUE("ruleValue"),
    @XmlEnumValue("standardError")
    STANDARD_ERROR("standardError"),
    @XmlEnumValue("transformedValue")
    TRANSFORMED_VALUE("transformedValue"),
    @XmlEnumValue("warning")
    WARNING("warning");
    private final String value;

    RESULTFEATURE(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static RESULTFEATURE fromValue(String v) {
        for (RESULTFEATURE c : RESULTFEATURE.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
