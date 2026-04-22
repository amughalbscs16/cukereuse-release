Feature: broken
  Scenario: unclosed doc string
    Given something
      """
      no closing marker
    Then this never parses
