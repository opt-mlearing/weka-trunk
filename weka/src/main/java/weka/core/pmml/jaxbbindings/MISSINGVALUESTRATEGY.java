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
 * <p>Java class for MISSING-VALUE-STRATEGY.
 *
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="MISSING-VALUE-STRATEGY">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="lastPrediction"/>
 *     &lt;enumeration value="nullPrediction"/>
 *     &lt;enumeration value="defaultChild"/>
 *     &lt;enumeration value="weightedConfidence"/>
 *     &lt;enumeration value="aggregateNodes"/>
 *     &lt;enumeration value="none"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 */
@XmlEnum
public enum MISSINGVALUESTRATEGY {

    @XmlEnumValue("aggregateNodes")
    AGGREGATE_NODES("aggregateNodes"),
    @XmlEnumValue("defaultChild")
    DEFAULT_CHILD("defaultChild"),
    @XmlEnumValue("lastPrediction")
    LAST_PREDICTION("lastPrediction"),
    @XmlEnumValue("none")
    NONE("none"),
    @XmlEnumValue("nullPrediction")
    NULL_PREDICTION("nullPrediction"),
    @XmlEnumValue("weightedConfidence")
    WEIGHTED_CONFIDENCE("weightedConfidence");
    private final String value;

    MISSINGVALUESTRATEGY(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static MISSINGVALUESTRATEGY fromValue(String v) {
        for (MISSINGVALUESTRATEGY c : MISSINGVALUESTRATEGY.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
