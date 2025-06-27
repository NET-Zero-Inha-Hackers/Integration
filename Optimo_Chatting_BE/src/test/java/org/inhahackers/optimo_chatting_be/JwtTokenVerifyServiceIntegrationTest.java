package org.inhahackers.optimo_chatting_be;

import org.inhahackers.optimo_chatting_be.exception.JwtAuthenticationException;
import org.inhahackers.optimo_chatting_be.service.JwtTokenVerifyService;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@SpringBootTest
class JwtTokenVerifyServiceIntegrationTest {

    @Autowired
    private JwtTokenVerifyService jwtTokenVerifyService;

    @ParameterizedTest
    @CsvSource({
            "this.is.invalid.token",
            "123456",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.payload.signature"
    })
    void validateToken_withInvalidToken_shouldThrowException(String invalidToken) {
        assertThatThrownBy(() -> jwtTokenVerifyService.validateToken(invalidToken))
                .isInstanceOf(JwtAuthenticationException.class);
    }
}