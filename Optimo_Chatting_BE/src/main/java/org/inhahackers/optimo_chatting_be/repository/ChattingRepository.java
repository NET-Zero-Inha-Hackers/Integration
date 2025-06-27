package org.inhahackers.optimo_chatting_be.repository;

import org.inhahackers.optimo_chatting_be.entity.Chatting;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ChattingRepository extends MongoRepository<Chatting, String> {
    List<Chatting> findAllByOwnerId(String ownerId);
    // chatList를 제외한 모든 필드 반환
    @Query(value = "{ 'ownerId': ?0 }", fields = "{ 'chatList': 0 }")
    List<Chatting> findAllByOwnerIdWithoutChatList(String ownerId);     // 다건 조회 (unique가 아닌 경우)
    void deleteByOwnerId(String ownerId);
}
