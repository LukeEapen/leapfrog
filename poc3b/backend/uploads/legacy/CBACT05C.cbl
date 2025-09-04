       IDENTIFICATION DIVISION.
       PROGRAM-ID. CALC-INTEREST.

       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT TXN-FILE ASSIGN TO 'TXN.DAT'
               ORGANIZATION IS SEQUENTIAL.

       DATA DIVISION.
       FILE SECTION.
       FD  TXN-FILE.
       01  TXN-RECORD.
           05  CARD-NUMBER            PIC X(16).
           05  TXN-DATE               PIC 9(8).
           05  TXN-AMOUNT             PIC S9(9)V99 COMP-3.
           05  BALANCE                PIC S9(9)V99 COMP-3.
           05  PREV-PMT-FULL          PIC X.

       WORKING-STORAGE SECTION.
       77  WS-EOF                     PIC X VALUE 'N'.
       77  WS-APR                     PIC S9(3)V99 COMP-3 VALUE 0.1999.
       77  WS-DAILY-RATE              PIC S9(5)V7 COMP-3.
       77  WS-INTEREST                PIC S9(9)V99 COMP-3.
       77  WS-DAYS-SINCE-TXN          PIC 9(3).

       01  WS-CURRENT-DATE.
           05  WS-CURR-YYYY           PIC 9(4).
           05  WS-CURR-MM             PIC 9(2).
           05  WS-CURR-DD             PIC 9(2).

       PROCEDURE DIVISION.
       MAIN-PROCEDURE.
           PERFORM INIT.
           PERFORM UNTIL WS-EOF = 'Y'
               READ TXN-FILE
                   AT END
                       MOVE 'Y' TO WS-EOF
                   NOT AT END
                       PERFORM CALCULATE-INTEREST
               END-READ
           END-PERFORM
           STOP RUN.

       INIT.
           ACCEPT WS-CURRENT-DATE FROM DATE YYYYMMDD
           COMPUTE WS-DAILY-RATE = WS-APR / 365.

       CALCULATE-INTEREST.
           COMPUTE WS-DAYS-SINCE-TXN =
               FUNCTION INTEGER-OF-DATE(WS-CURRENT-DATE)
             - FUNCTION INTEGER-OF-DATE(TXN-DATE)

           IF PREV-PMT-FULL = 'Y'
               MOVE ZERO TO WS-INTEREST
           ELSE
               COMPUTE WS-INTEREST =
                   BALANCE * WS-DAILY-RATE * WS-DAYS-SINCE-TXN
           END-IF

           DISPLAY 'CARD-NUM: ' CARD-NUMBER
           DISPLAY 'INTEREST : ' WS-INTEREST.