// DO NOT EDIT
// Generated by JFlex 1.8.2 http://jflex.de/
// source: src/main/java/weka/core/expressionlanguage/parser/Scanner.jflex

/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Scanner.java
 * Copyright (C) 2015 University of Waikato, Hamilton, New Zealand
 */

package weka.core.expressionlanguage.parser;

import java_cup.runtime.*;

import weka.core.expressionlanguage.core.SyntaxException;

/**
 * A lexical scanner for an expression language.
 * <p>
 * It emerged as a superset of the weka.core.mathematicalexpression package,
 * the weka.filers.unsupervised.instance.subsetbyexpression package and the
 * weka.core.AttributeExpression class.
 * <p>
 * Warning: This file (Scanner.java) has been auto generated by jflex.
 *
 * @author Benjamin Weber ( benweber at student dot ethz dot ch )
 * @version $Revision: 1000 $
 */

// See https://github.com/jflex-de/jflex/issues/222
@SuppressWarnings("FallThrough")
public class Scanner implements java_cup.runtime.Scanner {

    /**
     * This character denotes the end of file.
     */
    public static final int YYEOF = -1;

    /**
     * Initial size of the lookahead buffer.
     */
    private static final int ZZ_BUFFERSIZE = 16384;

    // Lexical states.
    public static final int YYINITIAL = 0;
    public static final int STRING1 = 2;
    public static final int STRING2 = 4;

    /**
     * ZZ_LEXSTATE[l] is the state in the DFA for the lexical state l
     * ZZ_LEXSTATE[l+1] is the state in the DFA for the lexical state l
     * at the beginning of a line
     * l is of the form l = 2*k, k a non negative integer
     */
    private static final int ZZ_LEXSTATE[] = {
            0, 0, 1, 1, 2, 2
    };

    /**
     * Top-level table for translating characters to character classes
     */
    private static final int[] ZZ_CMAP_TOP = zzUnpackcmap_top();

    private static final String ZZ_CMAP_TOP_PACKED_0 =
            "\1\0\25\u0100\1\u0200\11\u0100\1\u0300\17\u0100\1\u0400\247\u0100" +
                    "\10\u0500\u1020\u0100";

    private static int[] zzUnpackcmap_top() {
        int[] result = new int[4352];
        int offset = 0;
        offset = zzUnpackcmap_top(ZZ_CMAP_TOP_PACKED_0, offset, result);
        return result;
    }

    private static int zzUnpackcmap_top(String packed, int offset, int[] result) {
        int i = 0;       /* index in packed string  */
        int j = offset;  /* index in unpacked array */
        int l = packed.length();
        while (i < l) {
            int count = packed.charAt(i++);
            int value = packed.charAt(i++);
            do result[j++] = value; while (--count > 0);
        }
        return j;
    }


    /**
     * Second-level tables for translating characters to character classes
     */
    private static final int[] ZZ_CMAP_BLOCKS = zzUnpackcmap_blocks();

    private static final String ZZ_CMAP_BLOCKS_PACKED_0 =
            "\11\0\1\1\1\2\3\1\22\0\1\1\1\3\1\4" +
                    "\3\0\1\5\1\6\1\7\1\10\1\11\1\12\1\13" +
                    "\1\14\1\15\1\16\12\17\2\0\1\20\1\21\1\22" +
                    "\2\0\32\23\1\0\1\24\1\0\1\25\1\23\1\0" +
                    "\1\26\1\27\1\23\1\30\1\31\1\32\1\33\1\23" +
                    "\1\34\2\23\1\35\1\23\1\36\1\37\1\40\1\23" +
                    "\1\41\1\42\1\43\1\44\2\23\1\45\2\23\1\0" +
                    "\1\46\10\0\1\1\32\0\1\1\u01df\0\1\1\177\0" +
                    "\13\1\35\0\2\1\5\0\1\1\57\0\1\1\240\0" +
                    "\1\1\377\0\u0100\47";

    private static int[] zzUnpackcmap_blocks() {
        int[] result = new int[1536];
        int offset = 0;
        offset = zzUnpackcmap_blocks(ZZ_CMAP_BLOCKS_PACKED_0, offset, result);
        return result;
    }

    private static int zzUnpackcmap_blocks(String packed, int offset, int[] result) {
        int i = 0;       /* index in packed string  */
        int j = offset;  /* index in unpacked array */
        int l = packed.length();
        while (i < l) {
            int count = packed.charAt(i++);
            int value = packed.charAt(i++);
            do result[j++] = value; while (--count > 0);
        }
        return j;
    }

    /**
     * Translates DFA states to action switch labels.
     */
    private static final int[] ZZ_ACTION = zzUnpackAction();

    private static final String ZZ_ACTION_PACKED_0 =
            "\3\0\1\1\1\2\1\3\1\4\1\5\1\6\1\7" +
                    "\1\10\1\11\1\12\1\13\1\14\1\15\1\16\1\17" +
                    "\1\20\1\21\1\22\1\23\7\22\1\24\1\25\1\26" +
                    "\1\25\1\16\1\27\1\30\2\22\1\31\1\22\1\24" +
                    "\2\22\1\32\1\33\1\34\1\35\1\36\1\37\1\40" +
                    "\1\41\1\42\1\5\1\22\1\3\4\22\1\43\1\44" +
                    "\1\22\1\45";

    private static int[] zzUnpackAction() {
        int[] result = new int[63];
        int offset = 0;
        offset = zzUnpackAction(ZZ_ACTION_PACKED_0, offset, result);
        return result;
    }

    private static int zzUnpackAction(String packed, int offset, int[] result) {
        int i = 0;       /* index in packed string  */
        int j = offset;  /* index in unpacked array */
        int l = packed.length();
        while (i < l) {
            int count = packed.charAt(i++);
            int value = packed.charAt(i++);
            do result[j++] = value; while (--count > 0);
        }
        return j;
    }


    /**
     * Translates a state to a row index in the transition table
     */
    private static final int[] ZZ_ROWMAP = zzUnpackRowMap();

    private static final String ZZ_ROWMAP_PACKED_0 =
            "\0\0\0\50\0\120\0\170\0\170\0\170\0\170\0\170" +
                    "\0\170\0\170\0\170\0\170\0\170\0\170\0\170\0\170" +
                    "\0\240\0\310\0\170\0\360\0\u0118\0\170\0\u0140\0\u0168" +
                    "\0\u0190\0\u01b8\0\u01e0\0\u0208\0\u0230\0\170\0\170\0\170" +
                    "\0\u0258\0\u0280\0\170\0\170\0\u02a8\0\u02d0\0\u0118\0\u02f8" +
                    "\0\u0118\0\u0320\0\u0348\0\170\0\170\0\170\0\170\0\170" +
                    "\0\170\0\170\0\170\0\170\0\u0118\0\u0370\0\u0118\0\u0398" +
                    "\0\u03c0\0\u03e8\0\u0410\0\u0118\0\u0118\0\u0438\0\u0118";

    private static int[] zzUnpackRowMap() {
        int[] result = new int[63];
        int offset = 0;
        offset = zzUnpackRowMap(ZZ_ROWMAP_PACKED_0, offset, result);
        return result;
    }

    private static int zzUnpackRowMap(String packed, int offset, int[] result) {
        int i = 0;  /* index in packed string  */
        int j = offset;  /* index in unpacked array */
        int l = packed.length();
        while (i < l) {
            int high = packed.charAt(i++) << 16;
            result[j++] = high | packed.charAt(i++);
        }
        return j;
    }

    /**
     * The transition table of the DFA
     */
    private static final int[] ZZ_TRANS = zzUnpackTrans();

    private static final String ZZ_TRANS_PACKED_0 =
            "\1\4\2\5\1\6\1\7\1\10\1\11\1\12\1\13" +
                    "\1\14\1\15\1\16\1\17\1\4\1\20\1\21\1\22" +
                    "\1\23\1\24\1\25\1\4\1\26\1\27\3\25\1\30" +
                    "\1\25\1\31\1\25\1\32\1\33\1\25\1\34\1\25" +
                    "\1\35\2\25\1\36\1\4\2\37\1\4\1\37\1\40" +
                    "\17\37\1\41\22\37\1\4\2\37\1\4\3\37\1\40" +
                    "\15\37\1\41\22\37\1\4\65\0\1\42\1\0\1\21" +
                    "\51\0\1\43\47\0\1\44\45\0\1\25\3\0\1\25" +
                    "\2\0\20\25\21\0\1\25\3\0\1\25\2\0\10\25" +
                    "\1\45\7\25\21\0\1\25\3\0\1\25\2\0\1\46" +
                    "\17\25\21\0\1\25\3\0\1\25\2\0\14\25\1\47" +
                    "\3\25\21\0\1\25\3\0\1\25\2\0\11\25\1\50" +
                    "\6\25\21\0\1\25\3\0\1\25\2\0\13\25\1\51" +
                    "\4\25\21\0\1\25\3\0\1\25\2\0\3\25\1\52" +
                    "\14\25\21\0\1\25\3\0\1\25\2\0\13\25\1\53" +
                    "\4\25\2\0\2\54\1\0\1\54\1\55\1\54\1\56" +
                    "\15\54\1\57\2\54\1\60\2\54\1\61\3\54\1\62" +
                    "\2\54\1\63\1\54\1\64\3\54\20\0\1\42\47\0" +
                    "\1\25\3\0\1\25\2\0\2\25\1\65\15\25\21\0" +
                    "\1\25\3\0\1\25\2\0\7\25\1\66\10\25\21\0" +
                    "\1\25\3\0\1\25\2\0\15\25\1\67\2\25\21\0" +
                    "\1\25\3\0\1\25\2\0\5\25\1\70\12\25\21\0" +
                    "\1\25\3\0\1\25\2\0\16\25\1\71\1\25\21\0" +
                    "\1\25\3\0\1\25\2\0\14\25\1\72\3\25\21\0" +
                    "\1\25\3\0\1\25\2\0\3\25\1\73\14\25\21\0" +
                    "\1\25\3\0\1\25\2\0\3\25\1\74\14\25\21\0" +
                    "\1\25\3\0\1\25\2\0\3\25\1\75\14\25\21\0" +
                    "\1\25\3\0\1\25\2\0\17\25\1\76\21\0\1\25" +
                    "\3\0\1\25\2\0\12\25\1\77\5\25\2\0";

    private static int[] zzUnpackTrans() {
        int[] result = new int[1120];
        int offset = 0;
        offset = zzUnpackTrans(ZZ_TRANS_PACKED_0, offset, result);
        return result;
    }

    private static int zzUnpackTrans(String packed, int offset, int[] result) {
        int i = 0;       /* index in packed string  */
        int j = offset;  /* index in unpacked array */
        int l = packed.length();
        while (i < l) {
            int count = packed.charAt(i++);
            int value = packed.charAt(i++);
            value--;
            do result[j++] = value; while (--count > 0);
        }
        return j;
    }


    /**
     * Error code for "Unknown internal scanner error".
     */
    private static final int ZZ_UNKNOWN_ERROR = 0;
    /**
     * Error code for "could not match input".
     */
    private static final int ZZ_NO_MATCH = 1;
    /**
     * Error code for "pushback value was too large".
     */
    private static final int ZZ_PUSHBACK_2BIG = 2;

    /**
     * Error messages for {@link #ZZ_UNKNOWN_ERROR}, {@link #ZZ_NO_MATCH}, and
     * {@link #ZZ_PUSHBACK_2BIG} respectively.
     */
    private static final String ZZ_ERROR_MSG[] = {
            "Unknown internal scanner error",
            "Error: could not match input",
            "Error: pushback value was too large"
    };

    /**
     * ZZ_ATTRIBUTE[aState] contains the attributes of state {@code aState}
     */
    private static final int[] ZZ_ATTRIBUTE = zzUnpackAttribute();

    private static final String ZZ_ATTRIBUTE_PACKED_0 =
            "\3\0\15\11\2\1\1\11\2\1\1\11\7\1\3\11" +
                    "\2\1\2\11\7\1\11\11\13\1";

    private static int[] zzUnpackAttribute() {
        int[] result = new int[63];
        int offset = 0;
        offset = zzUnpackAttribute(ZZ_ATTRIBUTE_PACKED_0, offset, result);
        return result;
    }

    private static int zzUnpackAttribute(String packed, int offset, int[] result) {
        int i = 0;       /* index in packed string  */
        int j = offset;  /* index in unpacked array */
        int l = packed.length();
        while (i < l) {
            int count = packed.charAt(i++);
            int value = packed.charAt(i++);
            do result[j++] = value; while (--count > 0);
        }
        return j;
    }

    /**
     * Input device.
     */
    private java.io.Reader zzReader;

    /**
     * Current state of the DFA.
     */
    private int zzState;

    /**
     * Current lexical state.
     */
    private int zzLexicalState = YYINITIAL;

    /**
     * This buffer contains the current text to be matched and is the source of the {@link #yytext()}
     * string.
     */
    private char zzBuffer[] = new char[ZZ_BUFFERSIZE];

    /**
     * Text position at the last accepting state.
     */
    private int zzMarkedPos;

    /**
     * Current text position in the buffer.
     */
    private int zzCurrentPos;

    /**
     * Marks the beginning of the {@link #yytext()} string in the buffer.
     */
    private int zzStartRead;

    /**
     * Marks the last character in the buffer, that has been read from input.
     */
    private int zzEndRead;

    /**
     * Whether the scanner is at the end of file.
     *
     * @see #yyatEOF
     */
    private boolean zzAtEOF;

    /**
     * The number of occupied positions in {@link #zzBuffer} beyond {@link #zzEndRead}.
     *
     * <p>When a lead/high surrogate has been read from the input stream into the final
     * {@link #zzBuffer} position, this will have a value of 1; otherwise, it will have a value of 0.
     */
    private int zzFinalHighSurrogate = 0;

    /**
     * Number of newlines encountered up to the start of the matched text.
     */
    @SuppressWarnings("unused")
    private int yyline;

    /**
     * Number of characters from the last newline up to the start of the matched text.
     */
    @SuppressWarnings("unused")
    private int yycolumn;

    /**
     * Number of characters up to the start of the matched text.
     */
    @SuppressWarnings("unused")
    private long yychar;

    /**
     * Whether the scanner is currently at the beginning of a line.
     */
    @SuppressWarnings("unused")
    private boolean zzAtBOL = true;

    /**
     * Whether the user-EOF-code has already been executed.
     */
    @SuppressWarnings("unused")
    private boolean zzEOFDone;

    /* user code: */
    private StringBuilder string = new StringBuilder();

    private Symbol symbol(int type) {
        return new Symbol(type);
    }

    private Symbol symbol(int type, Object obj) {
        return new Symbol(type, obj);
    }


    /**
     * Creates a new scanner
     *
     * @param in the java.io.Reader to read input from.
     */
    public Scanner(java.io.Reader in) {
        this.zzReader = in;
    }

    /**
     * Translates raw input code points to DFA table row
     */
    private static int zzCMap(int input) {
        int offset = input & 255;
        return offset == input ? ZZ_CMAP_BLOCKS[offset] : ZZ_CMAP_BLOCKS[ZZ_CMAP_TOP[input >> 8] | offset];
    }

    /**
     * Refills the input buffer.
     *
     * @return {@code false} iff there was new input.
     * @throws java.io.IOException if any I/O-Error occurs
     */
    private boolean zzRefill() throws java.io.IOException {

        /* first: make room (if you can) */
        if (zzStartRead > 0) {
            zzEndRead += zzFinalHighSurrogate;
            zzFinalHighSurrogate = 0;
            System.arraycopy(zzBuffer, zzStartRead,
                    zzBuffer, 0,
                    zzEndRead - zzStartRead);

            /* translate stored positions */
            zzEndRead -= zzStartRead;
            zzCurrentPos -= zzStartRead;
            zzMarkedPos -= zzStartRead;
            zzStartRead = 0;
        }

        /* is the buffer big enough? */
        if (zzCurrentPos >= zzBuffer.length - zzFinalHighSurrogate) {
            /* if not: blow it up */
            char newBuffer[] = new char[zzBuffer.length * 2];
            System.arraycopy(zzBuffer, 0, newBuffer, 0, zzBuffer.length);
            zzBuffer = newBuffer;
            zzEndRead += zzFinalHighSurrogate;
            zzFinalHighSurrogate = 0;
        }

        /* fill the buffer with new input */
        int requested = zzBuffer.length - zzEndRead;
        int numRead = zzReader.read(zzBuffer, zzEndRead, requested);

        /* not supposed to occur according to specification of java.io.Reader */
        if (numRead == 0) {
            throw new java.io.IOException(
                    "Reader returned 0 characters. See JFlex examples/zero-reader for a workaround.");
        }
        if (numRead > 0) {
            zzEndRead += numRead;
            if (Character.isHighSurrogate(zzBuffer[zzEndRead - 1])) {
                if (numRead == requested) { // We requested too few chars to encode a full Unicode character
                    --zzEndRead;
                    zzFinalHighSurrogate = 1;
                } else {                    // There is room in the buffer for at least one more char
                    int c = zzReader.read();  // Expecting to read a paired low surrogate char
                    if (c == -1) {
                        return true;
                    } else {
                        zzBuffer[zzEndRead++] = (char) c;
                    }
                }
            }
            /* potentially more input available */
            return false;
        }

        /* numRead < 0 ==> end of stream */
        return true;
    }


    /**
     * Closes the input reader.
     *
     * @throws java.io.IOException if the reader could not be closed.
     */
    public final void yyclose() throws java.io.IOException {
        zzAtEOF = true; // indicate end of file
        zzEndRead = zzStartRead; // invalidate buffer

        if (zzReader != null) {
            zzReader.close();
        }
    }


    /**
     * Resets the scanner to read from a new input stream.
     *
     * <p>Does not close the old reader.
     *
     * <p>All internal variables are reset, the old input stream <b>cannot</b> be reused (internal
     * buffer is discarded and lost). Lexical state is set to {@code ZZ_INITIAL}.
     *
     * <p>Internal scan buffer is resized down to its initial length, if it has grown.
     *
     * @param reader The new input stream.
     */
    public final void yyreset(java.io.Reader reader) {
        zzReader = reader;
        zzEOFDone = false;
        yyResetPosition();
        zzLexicalState = YYINITIAL;
        if (zzBuffer.length > ZZ_BUFFERSIZE) {
            zzBuffer = new char[ZZ_BUFFERSIZE];
        }
    }

    /**
     * Resets the input position.
     */
    private final void yyResetPosition() {
        zzAtBOL = true;
        zzAtEOF = false;
        zzCurrentPos = 0;
        zzMarkedPos = 0;
        zzStartRead = 0;
        zzEndRead = 0;
        zzFinalHighSurrogate = 0;
        yyline = 0;
        yycolumn = 0;
        yychar = 0L;
    }


    /**
     * Returns whether the scanner has reached the end of the reader it reads from.
     *
     * @return whether the scanner has reached EOF.
     */
    public final boolean yyatEOF() {
        return zzAtEOF;
    }


    /**
     * Returns the current lexical state.
     *
     * @return the current lexical state.
     */
    public final int yystate() {
        return zzLexicalState;
    }


    /**
     * Enters a new lexical state.
     *
     * @param newState the new lexical state
     */
    public final void yybegin(int newState) {
        zzLexicalState = newState;
    }


    /**
     * Returns the text matched by the current regular expression.
     *
     * @return the matched text.
     */
    public final String yytext() {
        return new String(zzBuffer, zzStartRead, zzMarkedPos - zzStartRead);
    }


    /**
     * Returns the character at the given position from the matched text.
     *
     * <p>It is equivalent to {@code yytext().charAt(pos)}, but faster.
     *
     * @param position the position of the character to fetch. A value from 0 to {@code yylength()-1}.
     * @return the character at {@code position}.
     */
    public final char yycharat(int position) {
        return zzBuffer[zzStartRead + position];
    }


    /**
     * How many characters were matched.
     *
     * @return the length of the matched text region.
     */
    public final int yylength() {
        return zzMarkedPos - zzStartRead;
    }


    /**
     * Reports an error that occurred while scanning.
     *
     * <p>In a well-formed scanner (no or only correct usage of {@code yypushback(int)} and a
     * match-all fallback rule) this method will only be called with things that
     * "Can't Possibly Happen".
     *
     * <p>If this method is called, something is seriously wrong (e.g. a JFlex bug producing a faulty
     * scanner etc.).
     *
     * <p>Usual syntax/scanner level error handling should be done in error fallback rules.
     *
     * @param errorCode the code of the error message to display.
     */
    private static void zzScanError(int errorCode) throws SyntaxException {
        String message;
        try {
            message = ZZ_ERROR_MSG[errorCode];
        } catch (ArrayIndexOutOfBoundsException e) {
            message = ZZ_ERROR_MSG[ZZ_UNKNOWN_ERROR];
        }

        throw new SyntaxException(message);
    }


    /**
     * Pushes the specified amount of characters back into the input stream.
     *
     * <p>They will be read again by then next call of the scanning method.
     *
     * @param number the number of characters to be read again. This number must not be greater than
     *               {@link #yylength()}.
     */
    public void yypushback(int number) throws SyntaxException {
        if (number > yylength())
            zzScanError(ZZ_PUSHBACK_2BIG);

        zzMarkedPos -= number;
    }


    /**
     * Resumes scanning until the next regular expression is matched, the end of input is encountered
     * or an I/O-Error occurs.
     *
     * @return the next token.
     * @throws java.io.IOException if any I/O-Error occurs.
     */
    @Override
    public java_cup.runtime.Symbol next_token() throws java.io.IOException, SyntaxException {
        int zzInput;
        int zzAction;

        // cached fields:
        int zzCurrentPosL;
        int zzMarkedPosL;
        int zzEndReadL = zzEndRead;
        char[] zzBufferL = zzBuffer;

        int[] zzTransL = ZZ_TRANS;
        int[] zzRowMapL = ZZ_ROWMAP;
        int[] zzAttrL = ZZ_ATTRIBUTE;

        while (true) {
            zzMarkedPosL = zzMarkedPos;

            zzAction = -1;

            zzCurrentPosL = zzCurrentPos = zzStartRead = zzMarkedPosL;

            zzState = ZZ_LEXSTATE[zzLexicalState];

            // set up zzAction for empty match case:
            int zzAttributes = zzAttrL[zzState];
            if ((zzAttributes & 1) == 1) {
                zzAction = zzState;
            }


            zzForAction:
            {
                while (true) {

                    if (zzCurrentPosL < zzEndReadL) {
                        zzInput = Character.codePointAt(zzBufferL, zzCurrentPosL, zzEndReadL);
                        zzCurrentPosL += Character.charCount(zzInput);
                    } else if (zzAtEOF) {
                        zzInput = YYEOF;
                        break zzForAction;
                    } else {
                        // store back cached positions
                        zzCurrentPos = zzCurrentPosL;
                        zzMarkedPos = zzMarkedPosL;
                        boolean eof = zzRefill();
                        // get translated positions and possibly new buffer
                        zzCurrentPosL = zzCurrentPos;
                        zzMarkedPosL = zzMarkedPos;
                        zzBufferL = zzBuffer;
                        zzEndReadL = zzEndRead;
                        if (eof) {
                            zzInput = YYEOF;
                            break zzForAction;
                        } else {
                            zzInput = Character.codePointAt(zzBufferL, zzCurrentPosL, zzEndReadL);
                            zzCurrentPosL += Character.charCount(zzInput);
                        }
                    }
                    int zzNext = zzTransL[zzRowMapL[zzState] + zzCMap(zzInput)];
                    if (zzNext == -1) break zzForAction;
                    zzState = zzNext;

                    zzAttributes = zzAttrL[zzState];
                    if ((zzAttributes & 1) == 1) {
                        zzAction = zzState;
                        zzMarkedPosL = zzCurrentPosL;
                        if ((zzAttributes & 8) == 8) break zzForAction;
                    }

                }
            }

            // store back cached position
            zzMarkedPos = zzMarkedPosL;

            if (zzInput == YYEOF && zzStartRead == zzCurrentPos) {
                zzAtEOF = true;
                {
                    return new java_cup.runtime.Symbol(sym.EOF);
                }
            } else {
                switch (zzAction < 0 ? zzAction : ZZ_ACTION[zzAction]) {
                    case 1: {
                        throw new SyntaxException("Illegal character " + yytext() + "!");
                    }
                    // fall through
                    case 38:
                        break;
                    case 2: { /* ignore */
                    }
                    // fall through
                    case 39:
                        break;
                    case 3: {
                        return symbol(sym.NOT);
                    }
                    // fall through
                    case 40:
                        break;
                    case 4: {
                        yybegin(STRING1);
                        string.setLength(0);
                    }
                    // fall through
                    case 41:
                        break;
                    case 5: {
                        return symbol(sym.AND);
                    }
                    // fall through
                    case 42:
                        break;
                    case 6: {
                        yybegin(STRING2);
                        string.setLength(0);
                    }
                    // fall through
                    case 43:
                        break;
                    case 7: {
                        return symbol(sym.LPAREN);
                    }
                    // fall through
                    case 44:
                        break;
                    case 8: {
                        return symbol(sym.RPAREN);
                    }
                    // fall through
                    case 45:
                        break;
                    case 9: {
                        return symbol(sym.TIMES);
                    }
                    // fall through
                    case 46:
                        break;
                    case 10: {
                        return symbol(sym.PLUS);
                    }
                    // fall through
                    case 47:
                        break;
                    case 11: {
                        return symbol(sym.COMMA);
                    }
                    // fall through
                    case 48:
                        break;
                    case 12: {
                        return symbol(sym.MINUS);
                    }
                    // fall through
                    case 49:
                        break;
                    case 13: {
                        return symbol(sym.DIVISION);
                    }
                    // fall through
                    case 50:
                        break;
                    case 14: {
                        return symbol(sym.FLOAT, Double.valueOf(yytext()));
                    }
                    // fall through
                    case 51:
                        break;
                    case 15: {
                        return symbol(sym.LT);
                    }
                    // fall through
                    case 52:
                        break;
                    case 16: {
                        return symbol(sym.EQUAL);
                    }
                    // fall through
                    case 53:
                        break;
                    case 17: {
                        return symbol(sym.GT);
                    }
                    // fall through
                    case 54:
                        break;
                    case 18: {
                        return symbol(sym.IDENTIFIER, yytext());
                    }
                    // fall through
                    case 55:
                        break;
                    case 19: {
                        return symbol(sym.POW);
                    }
                    // fall through
                    case 56:
                        break;
                    case 20: {
                        return symbol(sym.OR);
                    }
                    // fall through
                    case 57:
                        break;
                    case 21: {
                        string.append(yytext());
                    }
                    // fall through
                    case 58:
                        break;
                    case 22: {
                        yybegin(YYINITIAL);
                        return symbol(sym.STRING, string.toString());
                    }
                    // fall through
                    case 59:
                        break;
                    case 23: {
                        return symbol(sym.LE);
                    }
                    // fall through
                    case 60:
                        break;
                    case 24: {
                        return symbol(sym.GE);
                    }
                    // fall through
                    case 61:
                        break;
                    case 25: {
                        return symbol(sym.IS);
                    }
                    // fall through
                    case 62:
                        break;
                    case 26: {
                        throw new SyntaxException("Invalid escape sequence '" + yytext() + "'!");
                    }
                    // fall through
                    case 63:
                        break;
                    case 27: {
                        string.append('\"');
                    }
                    // fall through
                    case 64:
                        break;
                    case 28: {
                        string.append('\'');
                    }
                    // fall through
                    case 65:
                        break;
                    case 29: {
                        string.append('\\');
                    }
                    // fall through
                    case 66:
                        break;
                    case 30: {
                        string.append('\b');
                    }
                    // fall through
                    case 67:
                        break;
                    case 31: {
                        string.append('\f');
                    }
                    // fall through
                    case 68:
                        break;
                    case 32: {
                        string.append('\n');
                    }
                    // fall through
                    case 69:
                        break;
                    case 33: {
                        string.append('\r');
                    }
                    // fall through
                    case 70:
                        break;
                    case 34: {
                        string.append('\t');
                    }
                    // fall through
                    case 71:
                        break;
                    case 35: {
                        return symbol(sym.BOOLEAN, true);
                    }
                    // fall through
                    case 72:
                        break;
                    case 36: {
                        return symbol(sym.BOOLEAN, false);
                    }
                    // fall through
                    case 73:
                        break;
                    case 37: {
                        return symbol(sym.REGEXP);
                    }
                    // fall through
                    case 74:
                        break;
                    default:
                        zzScanError(ZZ_NO_MATCH);
                }
            }
        }
    }


}
