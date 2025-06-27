package org.inhahackers.optimo_chatting_be.service;

import org.inhahackers.optimo_chatting_be.entity.Chatting;
import org.inhahackers.optimo_chatting_be.repository.ChattingRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ChattingService {

    private final ChattingRepository chattingRepository;

    public ChattingService(ChattingRepository chattingRepository) {
        this.chattingRepository = chattingRepository;
    }

    // 전체 채팅 리스트 조회
    public List<Chatting> findAllByOwnerIdWithoutChatList(String ownerId) {
        return chattingRepository.findAllByOwnerIdWithoutChatList(ownerId);
    }

    public List<Chatting> findAllByOwnerId(String ownerId) {
        return chattingRepository.findAllByOwnerId(ownerId);
    }

    // ID로 단일 채팅 조회
    public Chatting findById(String chattingId, String userId) {
        Chatting chatting = chattingRepository.findById(chattingId)
                .orElseThrow(() -> new IllegalArgumentException("Chatting not found with id: " + chattingId));

        if (!chatting.getOwnerId().equals(userId)) {
            throw new SecurityException("You are not authorized to delete this chatting.");
        }

        return chatting;
    }

    // ID로 채팅 삭제
    public void deleteById(String chattingId, String userId) {
        Chatting chatting = chattingRepository.findById(chattingId)
                .orElseThrow(() -> new IllegalArgumentException("Chatting not found with id: " + chattingId));

        if (!chatting.getOwnerId().equals(userId)) {
            throw new SecurityException("You are not authorized to delete this chatting.");
        }

        chattingRepository.deleteById(chattingId);
    }
}
