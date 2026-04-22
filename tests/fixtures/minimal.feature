Feature: Minimal feature with one scenario
  As a probe we want the simplest possible Given/When/Then.

  Scenario: greet
    Given a user named "Alice"
    When the user says hello
    Then the response is "Hi Alice"
