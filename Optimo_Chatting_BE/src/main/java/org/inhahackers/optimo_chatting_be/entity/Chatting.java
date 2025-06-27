package org.inhahackers.optimo_chatting_be.entity;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;
import lombok.Data;

import java.util.List;

@Data
@Document(collection = "chat_db")
public class Chatting {

    @Id
    private String chattingId;

    private @Indexed(unique = false) String ownerId;
    private String title;
    private String description;

    @Field("chatList")
    private List<ChatMessage> chatList;

    private long createdAt;
    private long modifiedAt;

    @Data
    public static class ChatMessage {
        private SenderType sender;  // enum 사용
        private String text;
        private long timestamp;
        private String model;
        private long use_estimate;
        private long llm_estimate;
    }
}
