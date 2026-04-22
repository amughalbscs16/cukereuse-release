Feature: Rule-based feature (Gherkin 6+)

  Rule: unauthenticated requests are rejected

    Scenario: missing token
      Given no auth token is provided
      When the client calls "/api/users"
      Then the response status is 401

  Rule: authenticated requests succeed

    Background:
      Given a valid token is provided

    Scenario: list users
      When the client calls "/api/users"
      Then the response status is 200
