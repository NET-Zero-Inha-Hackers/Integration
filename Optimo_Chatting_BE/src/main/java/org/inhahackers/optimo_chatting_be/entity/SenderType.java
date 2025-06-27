package org.inhahackers.optimo_chatting_be.entity;

import com.fasterxml.jackson.annotation.JsonValue;

public enum SenderType {
    USER("USER"),
    AI("AI");

    private final String value;

    SenderType(String value) {
        this.value = value;
    }

    @JsonValue // JSON 직렬화 시 문자열 값으로 표현됨
    public String getValue() {
        return value;
    }

    @Override
    public String toString() {
        return value;
    }
}
