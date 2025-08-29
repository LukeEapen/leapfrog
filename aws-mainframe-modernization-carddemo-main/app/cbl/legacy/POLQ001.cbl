       *================================================================*
       *  Program : POLQ001                                             *
       *  Purpose : Real-Time & Manual Policy Quoting & Issuance        *
       *            - Instant premium calc + dynamic risk assessment     *
       *            - Manual routing for complex cases                   *
       *            - On-demand policy doc event                         *
       *  Env     : CICS + DB2 + z/OS Connect (JSON over channels)      *
       *  Notes   : Replace STUBS with site adapters (HTTP/MQ).         *
       *================================================================*
       IDENTIFICATION DIVISION.
       PROGRAM-ID. POLQ001.
       AUTHOR.     ENTERPRISE-INS-PLATFORM-TEAM.
       INSTALLATION. US-INSURANCE-PLATFORM.
       DATE-WRITTEN. 2025-08-08.
       SECURITY.   INVOKED UNDER CICS; AUTH VIA APIM GW / SAF CLASS.

       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. IBM-Z15 WITH-DEBUGGING-MODE.

       DATA DIVISION.
       WORKING-STORAGE SECTION.

       *------------------------*
       *  CICS / DB2 Control    *
       *------------------------*
       01  WS-ABEND-CODE              PIC S9(09) COMP-4 VALUE 0.
       01  WS-RETCODE                 PIC S9(09) COMP-4 VALUE 0.
       01  WS-RESP                    PIC S9(09) COMP-4.
       01  WS-RESP2                   PIC S9(09) COMP-4.
       01  WS-CHANNEL                 PIC X(48) VALUE 'POLICY.CHANNEL'.
       01  WS-REQ-CONT                PIC X(48) VALUE 'REQUEST.JSON'.
       01  WS-RSP-CONT                PIC X(48) VALUE 'RESPONSE.JSON'.
       01  WS-AUD-CONT                PIC X(48) VALUE 'AUDIT.JSON'.

       EXEC SQL INCLUDE SQLCA END-EXEC.

       *------------------------*
       *  Timings (SLA)         *
       *------------------------*
       01  WS-START-TIME-STAMP        PIC S9(15) COMP-3.
       01  WS-END-TIME-STAMP          PIC S9(15) COMP-3.
       01  WS-ELAPSED-MS              PIC S9(9)  COMP-3.

       *------------------------*
       *  JSON Buffers          *
       *------------------------*
       01  WS-REQUEST-LEN             PIC S9(9) COMP-4 VALUE 0.
       01  WS-RESPONSE-LEN            PIC S9(9) COMP-4 VALUE 0.
       01  WS-AUDIT-LEN               PIC S9(9) COMP-4 VALUE 0.

       01  WS-REQUEST-JSON            PIC X(65535).
       01  WS-RESPONSE-JSON           PIC X(65535).
       01  WS-AUDIT-JSON              PIC X(16384).

       *------------------------*
       *  Parsed Request Model  *
       *  (Map to API contract) *
       *------------------------*
       01  REQ.
           05 REQ-CORRELATION-ID      PIC X(36).
           05 REQ-CHANNEL             PIC X(10).
           05 REQ-MODE                PIC X(10).  *> REALTIME | MANUAL
           05 REQ-PRODUCT             PIC X(10).  *> LIFE | HEALTH | PROPERTY

           05 REQ-APPLICANT.
              10 REQ-NAME             PIC X(60).
              10 REQ-DOB              PIC X(10).  *> YYYY-MM-DD
              10 REQ-SSN-LAST4        PIC X(4).
              10 REQ-ADDR1            PIC X(60).
              10 REQ-CITY             PIC X(30).
              10 REQ-STATE            PIC X(2).
              10 REQ-ZIP              PIC X(10).

           05 REQ-RISK-INPUTS.
              10 REQ-CREDIT-SCORE     PIC 9(3).
              10 REQ-BMI              PIC 9(3)V9(1).
              10 REQ-SMOKER           PIC X(1).   *> Y/N
              10 REQ-PROPERTY-ZIP     PIC X(10).
              10 REQ-PROPERTY-YEAR    PIC 9(4).

           05 REQ-COVERAGE.
              10 REQ-COV-LIMIT        PIC 9(9)V99.
              10 REQ-COV-TERM-MONTHS  PIC 9(4).
              10 REQ-DEDUCTIBLE       PIC 9(7)V99.

       *------------------------*
       *  Derived / External    *
       *------------------------*
       01  EXT-DATA.
           05 EXT-CREDIT-RISK         PIC 9(3).
           05 EXT-HEALTH-RISK         PIC 9(3).
           05 EXT-PROPERTY-RISK       PIC 9(3).
           05 EXT-HAZARD-SCORE        PIC 9(3).
           05 EXT-FRAUD-SCORE         PIC 9(3).

       *------------------------*
       *  Decision & Pricing    *
       *------------------------*
       01  DECISION-BLK.
           05 UW-NEEDED               PIC X(1) VALUE 'N'.
           05 UW-REASON-CODE          PIC X(10).
           05 QUOTE-ID                PIC X(18).
           05 POLICY-NUMBER           PIC X(20).
           05 PREMIUM-MONTHLY         PIC 9(9)V99.
           05 PREMIUM-ANNUAL          PIC 9(9)V99.
           05 TAX-AMT                 PIC 9(7)V99.
           05 FEES-AMT                PIC 9(7)V99.

       *------------------------*
       *  DB2 Host Vars         *
       *------------------------*
       01  HV-PRODUCT-CODE            PIC X(10).
       01  HV-RATE-TABLE-ID           PIC X(10).
       01  HV-STATE                   PIC X(2).
       01  HV-BASE-RATE               PIC 9(7)V999.
       01  HV-HAZARD-FACTOR           PIC 9(3)V99.
       01  HV-AGE-FACTOR              PIC 9(3)V99.

       * Sequence / IDs
       01  HV-QUOTE-ID                PIC X(18).
       01  HV-POLICY-NUM              PIC X(20).

       *------------------------*
       *  Constants / Limits    *
       *------------------------*
       78  MANUAL-CREDIT-THRESHOLD    VALUE 580.
       78  MANUAL-FRAUD-THRESHOLD     VALUE 700.
       78  MAX-RESPONSE-MS            VALUE 900.  *> target < 1s

       *------------------------*
       *  JSON PARSE/GEN STATE  *
       *------------------------*
       01  JSON-STATUS                PIC S9(9) COMP-4 VALUE 0.

       *------------------------*
       *  MQ/Kafka Event Payload *
       *------------------------*
       01  DOC-EVENT.
           05 DE-CORRELATION-ID       PIC X(36).
           05 DE-POLICY-NUMBER        PIC X(20).
           05 DE-DOC-TYPE             PIC X(20) VALUE 'POLICY_ISSUE'.
           05 DE-DELIVERY-CHANNEL     PIC X(10).  *> EMAIL/PORTAL
           05 DE-CUSTOMER-EMAIL       PIC X(80).

       LINKAGE SECTION.
       01  DFHEIBLK.
       01  DFHCOMMAREA.
          05  FILLER                  PIC X OCCURS 1 TO 32767
                                          DEPENDING ON EIBCALEN.

       PROCEDURE DIVISION.
       MAIN-ENTRY.
           EXEC CICS
                ASSIGN CHANNEL(WS-CHANNEL)
           END-EXEC

           PERFORM INIT-TIMING
           PERFORM RECEIVE-REQUEST
           PERFORM PARSE-REQUEST
              ON EXCEPTION
                 PERFORM BUILD-ERROR-RESPONSE
                 PERFORM SEND-RESPONSE
                 GOBACK
           END-PERFORM

           PERFORM VALIDATE-REQUEST
              ON EXCEPTION
                 PERFORM BUILD-ERROR-RESPONSE
                 PERFORM SEND-RESPONSE
                 GOBACK
           END-PERFORM

           PERFORM ENRICH-WITH-THIRD-PARTY
           PERFORM RISK-ASSESSMENT
           PERFORM RETRIEVE-RATES
           PERFORM CALCULATE-PREMIUM

           IF UW-NEEDED = 'Y'
              PERFORM PERSIST-QUOTE-PENDING-UW
              PERFORM BUILD-PENDING-UW-RESPONSE
           ELSE
              PERFORM ISSUE-POLICY
              PERFORM ENQUEUE-DOC-GEN
              PERFORM BUILD-ISSUED-RESPONSE
           END-IF

           PERFORM EMIT-AUDIT
           PERFORM SEND-RESPONSE
           GOBACK.

       *------------------------*
       *  INIT / TIMING         *
       *------------------------*
       INIT-TIMING.
           EXEC CICS ASKTIME ABSTIME(WS-START-TIME-STAMP) END-EXEC.

       *------------------------*
       *  RECEIVE REQUEST       *
       *------------------------*
       RECEIVE-REQUEST.
           EXEC CICS
                GET CONTAINER(WS-REQ-CONT)
                    CHANNEL(WS-CHANNEL)
                    INTO(WS-REQUEST-JSON)
                    FLENGTH(WS-REQUEST-LEN)
                    RESP(WS-RESP) RESP2(WS-RESP2)
           END-EXEC
           IF WS-RESP NOT = DFHRESP(NORMAL)
              MOVE 'Unable to read request' TO WS-RESPONSE-JSON
              MOVE FUNCTION LENGTH(WS-RESPONSE-JSON) TO WS-RESPONSE-LEN
              PERFORM SEND-RESPONSE
              GOBACK
           END-IF
           .

       *------------------------*
       *  PARSE JSON (IBM Ent COBOL) *
       *------------------------*
       PARSE-REQUEST.
           JSON PARSE WS-REQUEST-JSON
                INTO REQ
                WITH DETAIL
                ON EXCEPTION
                   MOVE 1 TO JSON-STATUS
                NOT ON EXCEPTION
                   MOVE 0 TO JSON-STATUS
           END-JSON
           IF JSON-STATUS NOT = 0
              RAISE EXCEPTION
           END-IF
           .

       *------------------------*
       *  VALIDATIONS           *
       *------------------------*
       VALIDATE-REQUEST.
           IF REQ-CORRELATION-ID = SPACES OR
              REQ-PRODUCT NOT = 'LIFE' AND
              REQ-PRODUCT NOT = 'HEALTH' AND
              REQ-PRODUCT NOT = 'PROPERTY'
                RAISE EXCEPTION
           END-IF

           IF REQ-MODE NOT = 'REALTIME' AND REQ-MODE NOT = 'MANUAL'
                RAISE EXCEPTION
           END-IF
           .

       *------------------------*
       *  THIRD-PARTY ENRICH    *
       *  (STUB: replace with HTTP or MQ adapters)                     *
       *------------------------*
       ENRICH-WITH-THIRD-PARTY.
           CALL 'EXTF01' USING REQ REQ-RISK-INPUTS EXT-DATA
                RETURNING WS-RETCODE.
           IF WS-RETCODE NOT = 0
              MOVE 650 TO EXT-FRAUD-SCORE       *> fallback conservative
              MOVE 200 TO EXT-HAZARD-SCORE
           END-IF
           .

       *------------------------*
       *  RISK ASSESSMENT       *
       *  (Rules: manual when high risk/complex)                       *
       *------------------------*
       RISK-ASSESSMENT.
           CALL 'SCOR01' USING REQ EXT-DATA DECISION-BLK
                RETURNING WS-RETCODE.
           IF WS-RETCODE NOT = 0
              * Simple in-line fallback rules
              IF REQ-RISK-INPUTS::REQ-CREDIT-SCORE < MANUAL-CREDIT-THRESHOLD
                    OR EXT-FRAUD-SCORE > MANUAL-FRAUD-THRESHOLD
                 MOVE 'Y' TO UW-NEEDED
                 MOVE 'RISK' TO UW-REASON-CODE
              END-IF
           END-IF
           .

       *------------------------*
       *  RATE RETRIEVAL (DB2)  *
       *------------------------*
       RETRIEVE-RATES.
           MOVE REQ-PRODUCT TO HV-PRODUCT-CODE.
           MOVE REQ-APPLICANT::REQ-STATE TO HV-STATE.

           EXEC SQL
             SELECT RATE_TABLE_ID, BASE_RATE, HAZARD_FACTOR
               INTO :HV-RATE-TABLE-ID, :HV-BASE-RATE, :HV-HAZARD-FACTOR
               FROM RATING_CONFIG
              WHERE PRODUCT_CODE = :HV-PRODUCT-CODE
                AND STATE        = :HV-STATE
           END-EXEC
           IF SQLCODE NOT = 0
              MOVE 1 TO WS-ABEND-CODE
              PERFORM BUILD-ERROR-RESPONSE
              PERFORM SEND-RESPONSE
              GOBACK
           END-IF

           * Example age factor query
           EXEC SQL
             SELECT AGE_FACTOR
               INTO :HV-AGE-FACTOR
               FROM AGE_FACTORS
              WHERE PRODUCT_CODE = :HV-PRODUCT-CODE
                AND AGE_YEARS = FLOOR(DAYS(CURRENT DATE)
                                 - DAYS(DATE(:REQ-APPLICANT::REQ-DOB))) / 365
           END-EXEC
           IF SQLCODE NOT = 0
              MOVE 1 TO WS-ABEND-CODE
              PERFORM BUILD-ERROR-RESPONSE
              PERFORM SEND-RESPONSE
              GOBACK
           END-IF
           .

       *------------------------*
       *  PREMIUM CALC          *
       *------------------------*
       CALCULATE-PREMIUM.
           CALL 'RATE01'
                USING REQ EXT-DATA
                      HV-BASE-RATE HV-HAZARD-FACTOR HV-AGE-FACTOR
                      DECISION-BLK
                RETURNING WS-RETCODE.
           IF WS-RETCODE NOT = 0
              * Fallback: base * hazard * age + fees/taxes
              COMPUTE PREMIUM-ANNUAL ROUNDED =
                      HV-BASE-RATE
                    * (1 + (HV-HAZARD-FACTOR / 100))
                    * (1 + (HV-AGE-FACTOR / 100)).
              COMPUTE FEES-AMT = 25.00
              COMPUTE TAX-AMT  = PREMIUM-ANNUAL * 0.015
              COMPUTE PREMIUM-MONTHLY ROUNDED =
                      (PREMIUM-ANNUAL + FEES-AMT + TAX-AMT) / 12
           END-IF

           IF UW-NEEDED = 'N' AND REQ-MODE = 'MANUAL'
              * Respect manual mode: route to UW anyway
              MOVE 'Y' TO UW-NEEDED
              MOVE 'MANUAL' TO UW-REASON-CODE
           END-IF
           .

       *------------------------*
       *  PERSIST QUOTE (Pending UW)
       *------------------------*
       PERSIST-QUOTE-PENDING-UW.
           EXEC SQL
             SELECT NEXT VALUE FOR QUOTE_SEQ INTO :HV-QUOTE-ID FROM SYSIBM.SYSDUMMY1
           END-EXEC
           MOVE HV-QUOTE-ID TO QUOTE-ID

           EXEC SQL
             INSERT INTO QUOTES
                (QUOTE_ID, CORRELATION_ID, PRODUCT, STATE, MONTHLY_PREM,
                 ANNUAL_PREM, TAX_AMT, FEES_AMT, UW_NEEDED, UW_REASON,
                 CREATED_TS)
             VALUES
                (:HV-QUOTE-ID, :REQ-CORRELATION-ID, :REQ-PRODUCT, :HV-STATE,
                 :PREMIUM-MONTHLY, :PREMIUM-ANNUAL, :TAX-AMT, :FEES-AMT,
                 'Y', :UW-REASON-CODE, CURRENT TIMESTAMP)
           END-EXEC
           .

       *------------------------*
       *  ISSUE POLICY (Auto)   *
       *------------------------*
       ISSUE-POLICY.
           EXEC SQL
             SELECT NEXT VALUE FOR POLICY_SEQ INTO :HV-POLICY-NUM
               FROM SYSIBM.SYSDUMMY1
           END-EXEC
           MOVE HV-POLICY-NUM TO POLICY-NUMBER

           EXEC SQL
             INSERT INTO POLICIES
               (POLICY_NUM, PRODUCT, STATE, CUST_NAME, COV_LIMIT,
                TERM_MONTHS, DEDUCTIBLE, PREMIUM_MONTHLY, PREMIUM_ANNUAL,
                TAX_AMT, FEES_AMT, STATUS, CREATED_TS)
             VALUES
               (:HV-POLICY-NUM, :REQ-PRODUCT, :HV-STATE, :REQ-APPLICANT::REQ-NAME,
                :REQ-COVERAGE::REQ-COV-LIMIT, :REQ-COVERAGE::REQ-COV-TERM-MONTHS,
                :REQ-COVERAGE::REQ-DEDUCTIBLE, :PREMIUM-MONTHLY, :PREMIUM-ANNUAL,
                :TAX-AMT, :FEES-AMT, 'ACTIVE', CURRENT TIMESTAMP)
           END-EXEC
           .

       *------------------------*
       *  DOCUMENT GEN EVENT    *
       *------------------------*
       ENQUEUE-DOC-GEN.
           MOVE REQ-CORRELATION-ID TO DE-CORRELATION-ID
           MOVE POLICY-NUMBER       TO DE-POLICY-NUMBER
           MOVE REQ-CHANNEL         TO DE-DELIVERY-CHANNEL
           MOVE SPACES              TO DE-CUSTOMER-EMAIL  *> fill when available

           CALL 'EVTQ01' USING DOC-EVENT RETURNING WS-RETCODE.
           *> EVTQ01: Site adapter — MQPUT to topic/queue used by doc service.
           .

       *------------------------*
       *  BUILD RESPONSES       *
       *------------------------*
       BUILD-PENDING-UW-RESPONSE.
           PERFORM STOP-TIMER
           STRING
             '{'
             '"correlationId":"', REQ-CORRELATION-ID, '",'
             '"status":"PENDING_UNDERWRITING",'
             '"reasonCode":"', UW-REASON-CODE, '",'
             '"quoteId":"', QUOTE-ID, '",'
             '"premium":{"monthly":', FUNCTION TRIM(FUNCTION NUMVAL-C(PREMIUM-MONTHLY)),
                          ',"annual":', FUNCTION TRIM(FUNCTION NUMVAL-C(PREMIUM-ANNUAL)),
                          ',"fees":',   FUNCTION TRIM(FUNCTION NUMVAL-C(FEES-AMT)),
                          ',"tax":',    FUNCTION TRIM(FUNCTION NUMVAL-C(TAX-AMT)), '},'
             '"slaMs":', FUNCTION TRIM(FUNCTION NUMVAL-C(WS-ELAPSED-MS))
             '}'
             DELIMITED BY SIZE INTO WS-RESPONSE-JSON
           END-STRING
           COMPUTE WS-RESPONSE-LEN = FUNCTION LENGTH(WS-RESPONSE-JSON)
           .

       BUILD-ISSUED-RESPONSE.
           PERFORM STOP-TIMER
           STRING
             '{'
             '"correlationId":"', REQ-CORRELATION-ID, '",'
             '"status":"ISSUED",'
             '"policyNumber":"', POLICY-NUMBER, '",'
             '"premium":{"monthly":', FUNCTION TRIM(FUNCTION NUMVAL-C(PREMIUM-MONTHLY)),
                          ',"annual":', FUNCTION TRIM(FUNCTION NUMVAL-C(PREMIUM-ANNUAL)),
                          ',"fees":',   FUNCTION TRIM(FUNCTION NUMVAL-C(FEES-AMT)),
                          ',"tax":',    FUNCTION TRIM(FUNCTION NUMVAL-C(TAX-AMT)), '},'
             '"slaMs":', FUNCTION TRIM(FUNCTION NUMVAL-C(WS-ELAPSED-MS))
             '}'
             DELIMITED BY SIZE INTO WS-RESPONSE-JSON
           END-STRING
           COMPUTE WS-RESPONSE-LEN = FUNCTION LENGTH(WS-RESPONSE-JSON)
           .

       BUILD-ERROR-RESPONSE.
           PERFORM STOP-TIMER
           STRING
             '{'
             '"status":"ERROR",'
             '"message":"Invalid request or internal error",'
             '"slaMs":', FUNCTION TRIM(FUNCTION NUMVAL-C(WS-ELAPSED-MS))
             '}'
             DELIMITED BY SIZE INTO WS-RESPONSE-JSON
           END-STRING
           COMPUTE WS-RESPONSE-LEN = FUNCTION LENGTH(WS-RESPONSE-JSON)
           .

       *------------------------*
       *  AUDIT / COMPLIANCE    *
       *------------------------*
       EMIT-AUDIT.
           STRING
             '{'
             '"cid":"', REQ-CORRELATION-ID, '",'
             '"product":"', REQ-PRODUCT, '",'
             '"mode":"', REQ-MODE, '",'
             '"uwNeeded":"', UW-NEEDED, '",'
             '"reason":"', UW-REASON-CODE, '",'
             '"elapsedMs":', FUNCTION TRIM(FUNCTION NUMVAL-C(WS-ELAPSED-MS)),
             '}'
             DELIMITED BY SIZE INTO WS-AUDIT-JSON
           END-STRING
           MOVE FUNCTION LENGTH(WS-AUDIT-JSON) TO WS-AUDIT-LEN

           CALL 'AUDL01' USING WS-AUDIT-JSON WS-AUDIT-LEN RETURNING WS-RETCODE.
           *> AUDL01: Site adapter — write to immutable audit store (e.g., Kafka topic + WORM).
           .

       *------------------------*
       *  SEND RESPONSE         *
       *------------------------*
       SEND-RESPONSE.
           EXEC CICS
                PUT CONTAINER(WS-RSP-CONT)
                    CHANNEL(WS-CHANNEL)
                    FROM(WS-RESPONSE-JSON)
                    FLENGTH(WS-RESPONSE-LEN)
                    RESP(WS-RESP) RESP2(WS-RESP2)
           END-EXEC
           .

       *------------------------*
       *  TIMING UTIL           *
       *------------------------*
       STOP-TIMER.
           EXEC CICS ASKTIME ABSTIME(WS-END-TIME-STAMP) END-EXEC
           COMPUTE WS-ELAPSED-MS =
                (WS-END-TIME-STAMP - WS-START-TIME-STAMP) / 1000.
           .

       END PROGRAM POLQ001.
