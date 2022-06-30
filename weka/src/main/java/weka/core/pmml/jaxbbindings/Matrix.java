//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.0-b52-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2013.12.20 at 12:48:21 PM GMT 
//


package weka.core.pmml.jaxbbindings;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for Matrix element declaration.
 *
 * <p>The following schema fragment specifies the expected content contained within this class.
 *
 * <pre>
 * &lt;element name="Matrix">
 *   &lt;complexType>
 *     &lt;complexContent>
 *       &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *         &lt;choice minOccurs="0">
 *           &lt;group ref="{http://www.dmg.org/PMML-4_1}NUM-ARRAY" maxOccurs="unbounded"/>
 *           &lt;element ref="{http://www.dmg.org/PMML-4_1}MatCell" maxOccurs="unbounded"/>
 *         &lt;/choice>
 *         &lt;attribute name="diagDefault" type="{http://www.dmg.org/PMML-4_1}REAL-NUMBER" />
 *         &lt;attribute name="kind" default="any">
 *           &lt;simpleType>
 *             &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *               &lt;enumeration value="diagonal"/>
 *               &lt;enumeration value="symmetric"/>
 *               &lt;enumeration value="any"/>
 *             &lt;/restriction>
 *           &lt;/simpleType>
 *         &lt;/attribute>
 *         &lt;attribute name="nbCols" type="{http://www.dmg.org/PMML-4_1}INT-NUMBER" />
 *         &lt;attribute name="nbRows" type="{http://www.dmg.org/PMML-4_1}INT-NUMBER" />
 *         &lt;attribute name="offDiagDefault" type="{http://www.dmg.org/PMML-4_1}REAL-NUMBER" />
 *       &lt;/restriction>
 *     &lt;/complexContent>
 *   &lt;/complexType>
 * &lt;/element>
 * </pre>
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
        "numarray",
        "matCell"
})
@XmlRootElement(name = "Matrix")
public class Matrix {

    @XmlElement(name = "Array", namespace = "http://www.dmg.org/PMML-4_1", required = true)
    protected List<ArrayType> numarray;
    @XmlElement(name = "MatCell", namespace = "http://www.dmg.org/PMML-4_1", required = true)
    protected List<MatCell> matCell;
    @XmlAttribute
    protected Double diagDefault;
    @XmlAttribute
    protected String kind;
    @XmlAttribute
    protected BigInteger nbCols;
    @XmlAttribute
    protected BigInteger nbRows;
    @XmlAttribute
    protected Double offDiagDefault;

    /**
     * Gets the value of the numarray property.
     *
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the numarray property.
     *
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getNUMARRAY().add(newItem);
     * </pre>
     *
     *
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ArrayType }
     */
    public List<ArrayType> getNUMARRAY() {
        if (numarray == null) {
            numarray = new ArrayList<ArrayType>();
        }
        return this.numarray;
    }

    /**
     * Gets the value of the matCell property.
     *
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the matCell property.
     *
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getMatCell().add(newItem);
     * </pre>
     *
     *
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link MatCell }
     */
    public List<MatCell> getMatCell() {
        if (matCell == null) {
            matCell = new ArrayList<MatCell>();
        }
        return this.matCell;
    }

    /**
     * Gets the value of the diagDefault property.
     *
     * @return possible object is
     * {@link Double }
     */
    public Double getDiagDefault() {
        return diagDefault;
    }

    /**
     * Sets the value of the diagDefault property.
     *
     * @param value allowed object is
     *              {@link Double }
     */
    public void setDiagDefault(Double value) {
        this.diagDefault = value;
    }

    /**
     * Gets the value of the kind property.
     *
     * @return possible object is
     * {@link String }
     */
    public String getKind() {
        if (kind == null) {
            return "any";
        } else {
            return kind;
        }
    }

    /**
     * Sets the value of the kind property.
     *
     * @param value allowed object is
     *              {@link String }
     */
    public void setKind(String value) {
        this.kind = value;
    }

    /**
     * Gets the value of the nbCols property.
     *
     * @return possible object is
     * {@link BigInteger }
     */
    public BigInteger getNbCols() {
        return nbCols;
    }

    /**
     * Sets the value of the nbCols property.
     *
     * @param value allowed object is
     *              {@link BigInteger }
     */
    public void setNbCols(BigInteger value) {
        this.nbCols = value;
    }

    /**
     * Gets the value of the nbRows property.
     *
     * @return possible object is
     * {@link BigInteger }
     */
    public BigInteger getNbRows() {
        return nbRows;
    }

    /**
     * Sets the value of the nbRows property.
     *
     * @param value allowed object is
     *              {@link BigInteger }
     */
    public void setNbRows(BigInteger value) {
        this.nbRows = value;
    }

    /**
     * Gets the value of the offDiagDefault property.
     *
     * @return possible object is
     * {@link Double }
     */
    public Double getOffDiagDefault() {
        return offDiagDefault;
    }

    /**
     * Sets the value of the offDiagDefault property.
     *
     * @param value allowed object is
     *              {@link Double }
     */
    public void setOffDiagDefault(Double value) {
        this.offDiagDefault = value;
    }

}
