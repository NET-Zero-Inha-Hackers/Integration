package org.inhahackers.optimo_chatting_be.service;

import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import jakarta.annotation.PostConstruct;
import org.inhahackers.optimo_chatting_be.exception.JwtAuthenticationException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.security.Key;

@Service
public class JwtTokenVerifyService {

    @Value("${jwt.secret}")
    private String secret;

    @Value("${jwt.access-token-validity-in-seconds}")
    private long tokenValidityInSeconds;

    private Key key;

    @PostConstruct
    public void init() {
        this.key = Keys.hmacShaKeyFor(secret.getBytes());
    }

    public Jws<Claims> validateToken(String token) {
        try {
            return Jwts.parserBuilder()
                    .setSigningKey(key)
                    .build()
                    .parseClaimsJws(token);
        } catch (JwtException e) {
            throw new JwtAuthenticationException("Invalid or expired JWT token", e);
        }
    }


    public Long extractUserId(String token) {
        return Long.parseLong(validateToken(token).getBody().getSubject());
    }

    public String extractEmail(String token) {
        return validateToken(token).getBody().get("email", String.class);
    }

    public String extractRole(String token) {
        return validateToken(token).getBody().get("role", String.class);
    }
}
