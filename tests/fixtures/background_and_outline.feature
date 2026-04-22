@smoke @api
Feature: Exercise Background + Scenario Outline + tables + doc strings

  Background:
    Given the service is running
    And the user is authenticated

  @happy
  Scenario: straightforward login
    When the user submits valid credentials
    Then the session is created

  Scenario Outline: login with <role>
    When the user logs in as "<role>"
    Then the dashboard for "<role>" is shown

    Examples:
      | role    |
      | admin   |
      | editor  |
      | viewer  |

  Scenario: multiline payload
    When the client posts the payload
      """
      {
        "name": "Bob",
        "age": 42
      }
      """
    Then the server responds with 201

  Scenario: payload with table
    When the client posts users
      | name  | email             |
      | Alice | alice@example.com |
      | Bob   | bob@example.com   |
    Then the response lists the users
